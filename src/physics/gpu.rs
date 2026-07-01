//! GPU compute foundation for the galaxy N-body (Barnes–Hut on the GPU).
//!
//! The plan is to keep the particles resident in GPU buffers and do the whole step
//! — build a tree, walk it for the forces, integrate — in compute shaders, so the
//! data never leaves the card and the point renderer can read the positions
//! directly. That is a lot of moving parts, so each one is validated the same way
//! the CPU octree is: run the kernel, read the buffer back, and compare against a
//! plain CPU reference. Compute needs no window, so those checks run headless in
//! the test suite.
//!
//! The pipeline is complete: bounding box, Morton codes, a bitonic sort, the Karras
//! LBVH build, a bottom-up mass/COM/box refit, and a per-particle tree walk — each a
//! compute kernel with its own CPU-reference test. [`GpuNBody`] ties them together
//! into a fully resident leapfrog: one command submission per step, with only the
//! positions copied back each frame to draw. See `docs/gpu-nbody.md` for the whole
//! story. The standalone `*_gpu` functions (which read each stage back) remain as the
//! validated reference the resident stepper is built from.
#![allow(dead_code)]

use glam::Vec3;
use wgpu::util::DeviceExt;

/// Spread the low 10 bits of `v` out into every third bit (0..30).
///
/// What: turns `abc…` into `a..b..c..`, the building block of a 3-D Morton code.
/// How/why: interleaving three axes' spread bits gives the Z-order (Morton) key,
/// so points close in space get close keys — the ordering an LBVH is built on. The
/// magic-constant shifts are the standard branch-free bit spread.
/// Units: none (bit twiddling).
pub fn expand_bits(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x0300_00FF;
    x = (x | (x << 8)) & 0x0300_F00F;
    x = (x | (x << 4)) & 0x030C_30C3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// The 30-bit Morton code of a point inside the bounding box.
///
/// What: a Z-order key; nearby points get nearby keys.
/// How/why: normalise the point into the unit cube with `(p − lo)·inv`, quantise
/// each axis to 10 bits (0..1023), then interleave with [`expand_bits`]. Written to
/// match the GPU kernel bit-for-bit so the two can be checked against each other.
/// Units: `p`/`lo` a length; `inv = 1/(hi − lo)` per axis.
pub fn morton_code(p: Vec3, lo: Vec3, inv: Vec3) -> u32 {
    let u = ((p - lo) * inv).clamp(Vec3::ZERO, Vec3::splat(0.999_999));
    let xi = (u.x * 1024.0) as u32;
    let yi = (u.y * 1024.0) as u32;
    let zi = (u.z * 1024.0) as u32;
    expand_bits(xi) | (expand_bits(yi) << 1) | (expand_bits(zi) << 2)
}

/// The WGSL Morton-code kernel (one thread per particle).
const MORTON_SHADER: &str = r#"
struct Params { lo: vec4<f32>, inv: vec4<f32> };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> codes: array<u32>;

fn expand_bits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u))  & 0x0300F00Fu;
    x = (x | (x << 4u))  & 0x030C30C3u;
    x = (x | (x << 2u))  & 0x09249249u;
    return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&positions)) { return; }
    let u = clamp((positions[i].xyz - params.lo.xyz) * params.inv.xyz,
                  vec3<f32>(0.0), vec3<f32>(0.999999));
    let xi = u32(u.x * 1024.0);
    let yi = u32(u.y * 1024.0);
    let zi = u32(u.z * 1024.0);
    codes[i] = expand_bits(xi) | (expand_bits(yi) << 1u) | (expand_bits(zi) << 2u);
}
"#;

/// Compute the Morton codes of `positions` on the GPU and read them back.
///
/// What: returns one 30-bit Z-order key per particle.
/// How/why: uploads the positions (packed as `vec4`, xyz used) and the box
/// parameters, dispatches the [`MORTON_SHADER`] one thread per particle, then maps
/// the result buffer back. This one-shot form is for building and testing the
/// kernel; later phases fold it into the persistent GPU pipeline.
/// Units: `positions`/`lo` a length; `inv = 1/(hi − lo)` per axis.
pub fn compute_morton_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions: &[[f32; 4]],
    lo: Vec3,
    inv: Vec3,
) -> Vec<u32> {
    let n = positions.len();
    let params = [lo.x, lo.y, lo.z, 0.0, inv.x, inv.y, inv.z, 0.0];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("morton params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("morton positions"),
        contents: bytemuck::cast_slice(positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let codes_bytes = (n * std::mem::size_of::<u32>()) as u64;
    let codes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morton codes"),
        size: codes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morton readback"),
        size: codes_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("morton"),
        source: wgpu::ShaderSource::Wgsl(MORTON_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("morton layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morton bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: codes_buf.as_entire_binding(),
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("morton pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("morton pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("morton"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64).max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&codes_buf, 0, &readback, 0, codes_bytes.max(4));
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data)[..n].to_vec();
    out
}

/// The WGSL bitonic compare-exchange kernel (one thread per element).
///
/// Sorts by the pair `(key, val)` lexicographically, so equal Morton codes are
/// ordered by particle index — exactly the total order the LBVH build needs.
const BITONIC_SHADER: &str = r#"
struct P { j: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let partner = i ^ p.j;
    if (partner <= i) { return; }              // let the lower index do the work
    let ascending = (i & p.k) == 0u;           // direction of this bitonic block
    let ki = keys[i]; let kp = keys[partner];
    let vi = vals[i]; let vp = vals[partner];
    // Lexicographic (key, val) comparison.
    let a_gt = (ki > kp) || (ki == kp && vi > vp);
    let a_lt = (ki < kp) || (ki == kp && vi < vp);
    let do_swap = select(a_lt, a_gt, ascending);
    if (do_swap) {
        keys[i] = kp; keys[partner] = ki;
        vals[i] = vp; vals[partner] = vi;
    }
}
"#;

/// Sort `(keys, vals)` pairs on the GPU by `(key, val)` and read them back.
///
/// What: returns the pairs sorted ascending by key, ties broken by val.
/// How/why: a **bitonic sorting network**. We pad the array up to a power of two
/// with `0xFFFFFFFF` sentinels (which sort to the end), then run the fixed sequence
/// of compare-exchange sub-passes: for block size `k = 2,4,…,M` and stride
/// `j = k/2,…,1`, every element `i` compares with `i ^ j` and swaps to make its
/// block bitonic, ascending or descending per `(i & k)`. Each sub-pass is one
/// compute dispatch; the parameters `(j, k)` are fed through a uniform buffer with a
/// dynamic offset, so the whole sort is a single command submission. The order is
/// data-independent, so it is fully deterministic.
/// Principle: a bitonic network sorts any input in `O(N·log²N)` compare-exchanges,
/// all independent within a sub-pass — ideal for the GPU.
/// Units: none.
pub fn bitonic_sort_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    keys: &[u32],
    vals: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    assert_eq!(n, vals.len());
    if n <= 1 {
        return (keys.to_vec(), vals.to_vec());
    }
    let m = n.next_power_of_two();

    // Pad up to M with sentinels that sort to the very end.
    let mut kpad = keys.to_vec();
    let mut vpad = vals.to_vec();
    kpad.resize(m, u32::MAX);
    vpad.resize(m, u32::MAX);

    let key_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sort keys"),
        contents: bytemuck::cast_slice(&kpad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let val_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sort vals"),
        contents: bytemuck::cast_slice(&vpad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Build the (j, k) schedule and pack one 256-byte-aligned uniform block each,
    // so we can select the current sub-pass with a dynamic offset.
    const STRIDE: usize = 256; // safe uniform dynamic-offset alignment
    let mut schedule: Vec<(u32, u32)> = Vec::new();
    let mut k = 2usize;
    while k <= m {
        let mut j = k / 2;
        while j >= 1 {
            schedule.push((j as u32, k as u32));
            j /= 2;
        }
        k *= 2;
    }
    let mut params = vec![0u8; schedule.len() * STRIDE];
    for (idx, &(j, k)) in schedule.iter().enumerate() {
        let base = idx * STRIDE;
        params[base..base + 4].copy_from_slice(&j.to_le_bytes());
        params[base + 4..base + 8].copy_from_slice(&k.to_le_bytes());
        params[base + 8..base + 12].copy_from_slice(&(m as u32).to_le_bytes());
    }
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sort params"),
        contents: &params,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bitonic"),
        source: wgpu::ShaderSource::Wgsl(BITONIC_SHADER.into()),
    });
    let mut ulayout = uniform_entry(0);
    ulayout.ty = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: std::num::NonZeroU64::new(16),
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sort layout"),
        entries: &[ulayout, storage_entry(1, false), storage_entry(2, false)],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sort bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: key_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: val_buf.as_entire_binding(),
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sort pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bitonic pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let groups = (m as u32).div_ceil(64);
    // One compute pass per sub-pass, so each fully finishes (and its writes are
    // visible) before the next reads the buffers.
    for idx in 0..schedule.len() {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bitonic pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[(idx * STRIDE) as u32]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    // Read back only the first N (the sorted real entries; padding is at the end).
    let n_bytes = (n * 4) as u64;
    let kback = readback_buffer(device, n_bytes);
    let vback = readback_buffer(device, n_bytes);
    encoder.copy_buffer_to_buffer(&key_buf, 0, &kback, 0, n_bytes);
    encoder.copy_buffer_to_buffer(&val_buf, 0, &vback, 0, n_bytes);
    queue.submit(Some(encoder.finish()));

    let sorted_keys = map_u32(device, &kback, n);
    let sorted_vals = map_u32(device, &vback, n);
    (sorted_keys, sorted_vals)
}

/// The WGSL LBVH structure kernel (Karras 2012): one thread per internal node.
///
/// Node ids: internal nodes `0..n-1`, leaves `n-1..2n-1` (leaf `k` is id `n-1+k`);
/// the root is internal node 0. Each internal node finds the range of sorted
/// particles it covers and its split point from `δ` — the length of the common bit
/// prefix of neighbouring codes (with the array *index* as tiebreaker for equal
/// codes, which is why the sort ordered by `(code, index)`).
const LBVH_SHADER: &str = r#"
struct Uni { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> lft: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgt: array<u32>;
@group(0) @binding(4) var<storage, read_write> par: array<u32>;

// Length of the common prefix of the keys at positions i and j (−1 if j is off the
// end). Equal codes fall back to the indices, so every pair has a definite order.
fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) { return -1; }
    let a = codes[u32(i)];
    let b = codes[u32(j)];
    if (a == b) {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(u.n);
    let i = i32(gid.x);
    if (i >= n - 1) { return; }

    // Direction of the range: toward whichever neighbour shares a longer prefix.
    let dr = delta(i, i + 1, n);
    let dl = delta(i, i - 1, n);
    var d = 1;
    if (dr < dl) { d = -1; }
    let delta_min = delta(i, i - d, n);

    // Grow an upper bound on the range length, then binary-search the exact length.
    var l_max = 2;
    while (delta(i, i + l_max * d, n) > delta_min) { l_max = l_max * 2; }
    var l = 0;
    var t = l_max / 2;
    while (t >= 1) {
        if (delta(i, i + (l + t) * d, n) > delta_min) { l = l + t; }
        t = t / 2;
    }
    let j = i + l * d;
    let delta_node = delta(i, j, n);

    // Binary-search the split position within the range.
    var s = 0;
    var dv = 2;
    loop {
        let tt = (l + dv - 1) / dv;               // ceil(l / dv)
        if (delta(i, i + (s + tt) * d, n) > delta_node) { s = s + tt; }
        if (tt <= 1) { break; }
        dv = dv * 2;
    }
    let gamma = i + s * d + min(d, 0);

    let first = min(i, j);
    let last = max(i, j);
    // A child is a leaf when its side of the split is a single element.
    var lc = u32(gamma);
    if (gamma == first) { lc = u32(n - 1 + gamma); }
    var rc = u32(gamma + 1);
    if (gamma + 1 == last) { rc = u32(n - 1 + gamma + 1); }

    lft[u32(i)] = lc;
    rgt[u32(i)] = rc;
    par[lc] = u32(i);
    par[rc] = u32(i);
}
"#;

/// Sentinel "no node" value (the root's parent).
pub const NO_NODE: u32 = 0xFFFF_FFFF;

/// Build the LBVH structure on the GPU from sorted Morton codes.
///
/// What: returns `(left, right, parent)` — for each of the `n-1` internal nodes its
/// two child ids, and for all `2n-1` nodes its parent (the root's is [`NO_NODE`]).
/// How/why: runs the Karras kernel, one thread per internal node, then reads the
/// three arrays back. Node ids: internal `0..n-1`, leaves `n-1..2n-1`.
/// Units: none (indices).
pub fn build_lbvh_structure_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    codes: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = codes.len();
    assert!(n >= 2, "need at least two particles");
    let internal = n - 1;
    let total = 2 * n - 1;

    let uni = [n as u32, 0, 0, 0];
    let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lbvh uni"),
        contents: bytemuck::cast_slice(&uni),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let codes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lbvh codes"),
        contents: bytemuck::cast_slice(codes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rw = |bytes: u64, label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let left_buf = rw((internal * 4) as u64, "lbvh left");
    let right_buf = rw((internal * 4) as u64, "lbvh right");
    // Parent starts all-sentinel; every node but the root gets one written.
    let parent_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lbvh parent"),
        contents: bytemuck::cast_slice(&vec![NO_NODE; total]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lbvh"),
        source: wgpu::ShaderSource::Wgsl(LBVH_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lbvh layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
        ],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lbvh bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uni_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: codes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: left_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: right_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: parent_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lbvh pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("lbvh pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lbvh"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((internal as u32).div_ceil(64), 1, 1);
    }
    let lback = readback_buffer(device, (internal * 4) as u64);
    let rback = readback_buffer(device, (internal * 4) as u64);
    let pback = readback_buffer(device, (total * 4) as u64);
    encoder.copy_buffer_to_buffer(&left_buf, 0, &lback, 0, (internal * 4) as u64);
    encoder.copy_buffer_to_buffer(&right_buf, 0, &rback, 0, (internal * 4) as u64);
    encoder.copy_buffer_to_buffer(&parent_buf, 0, &pback, 0, (total * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let left = map_u32(device, &lback, internal);
    let right = map_u32(device, &rback, internal);
    let parent = map_u32(device, &pback, total);
    (left, right, parent)
}

const AGGREGATE_SHADER: &str = r#"
struct Uni { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var<storage, read> lft: array<u32>;
@group(0) @binding(2) var<storage, read> rgt: array<u32>;
@group(0) @binding(3) var<storage, read_write> node_mass: array<f32>;
@group(0) @binding(4) var<storage, read_write> node_com: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> done_in: array<u32>;
@group(0) @binding(6) var<storage, read_write> done_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> node_lo: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> node_hi: array<vec4<f32>>;

// One level of the bottom-up refit. Runs once per pass; every pass lifts the
// "finished" frontier up by one level. A node combines only when BOTH children were
// finished in an EARLIER pass, so their writes (from that earlier submit) are already
// visible — that pass boundary is what makes this safe without relying on GPU atomics
// or a cross-workgroup memory model. Each node gets its total mass, centre of mass,
// and the axis-aligned box enclosing its part (the box gives its size for opening).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n - 1u) { return; }             // internal nodes are 0 … n-2
    if (done_in[i] == 1u) { done_out[i] = 1u; return; }   // already finished: carry on
    let l = lft[i];
    let r = rgt[i];
    if (done_in[l] == 0u || done_in[r] == 0u) {           // a child isn't ready yet
        done_out[i] = 0u;
        return;
    }
    let ml = node_mass[l];
    let mr = node_mass[r];
    let m = ml + mr;
    let c = (ml * node_com[l].xyz + mr * node_com[r].xyz) / m;  // mass-weighted centre
    node_mass[i] = m;
    node_com[i] = vec4<f32>(c, m);
    node_lo[i] = vec4<f32>(min(node_lo[l].xyz, node_lo[r].xyz), 0.0);
    node_hi[i] = vec4<f32>(max(node_hi[l].xyz, node_hi[r].xyz), 0.0);
    done_out[i] = 1u;
}
"#;

/// Aggregate each tree node's total mass and centre of mass, bottom-up, on the GPU.
///
/// The per-node arrays this refit produces: `(mass, com, box_lo, box_hi)`, one entry
/// per node (`.w` lanes unused except that `com.w` also carries the mass).
type NodeArrays = (Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<[f32; 4]>);

/// What: given the LBVH structure and the per-leaf masses/positions (in sorted
/// order), returns `(node_mass, node_com, node_lo, node_hi)` for all `2n-1` nodes —
/// leaves hold their own particle, internal nodes the combined mass, centre of mass,
/// and the axis-aligned box enclosing their part (which gives their size for the
/// opening test).
/// How/why: the tree is filled in **levels**. Leaves start "finished"; each pass, an
/// internal node whose two children were finished in an earlier pass sets its own
/// `M = m₁+m₂`, `C = (m₁·c₁+m₂·c₂)/M`, and box = union of the children's boxes, then
/// marks itself finished. The "finished" flags are double-buffered
/// (`done_in`/`done_out`, swapped each pass) so a node is never combined from a child
/// produced in the very same pass. We keep dispatching until the root is finished.
/// Each pass is its own submit, and that boundary is what guarantees a child's writes
/// are visible before its parent reads them — the robust alternative to a single-pass
/// atomic walk-up, whose cross-workgroup reads are not covered by WGSL's memory model.
/// The physics: a node's centre of mass is the mass-weighted average of its parts —
/// exactly what Barnes–Hut treats as one distant body when it "lumps" a node.
/// Units: masses in solar masses, positions/COM/box in AU (whatever the leaves use).
pub fn aggregate_nodes_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    left: &[u32],
    right: &[u32],
    parent: &[u32],
    leaf_mass: &[f32],
    leaf_com: &[[f32; 4]],
) -> NodeArrays {
    let n = leaf_mass.len();
    assert!(n >= 2, "need at least two particles");
    assert_eq!(left.len(), n - 1);
    assert_eq!(right.len(), n - 1);
    assert_eq!(parent.len(), 2 * n - 1);
    assert_eq!(leaf_com.len(), n);
    let internal = n - 1;
    let total = 2 * n - 1;

    // Seed the node arrays on the CPU: leaf k lives at node id (n-1)+k, internal
    // nodes start at zero and are overwritten before they are ever read. A leaf's
    // box is the single point of its particle.
    let mut mass_init = vec![0.0f32; total];
    let mut com_init = vec![[0.0f32; 4]; total];
    let mut lo_init = vec![[0.0f32; 4]; total];
    let mut hi_init = vec![[0.0f32; 4]; total];
    for k in 0..n {
        mass_init[internal + k] = leaf_mass[k];
        com_init[internal + k] = [leaf_com[k][0], leaf_com[k][1], leaf_com[k][2], leaf_mass[k]];
        lo_init[internal + k] = [leaf_com[k][0], leaf_com[k][1], leaf_com[k][2], 0.0];
        hi_init[internal + k] = [leaf_com[k][0], leaf_com[k][1], leaf_com[k][2], 0.0];
    }
    // "Finished" flags: leaves start finished, internal nodes not. Two identical
    // copies so we can ping-pong (leaf entries are never written, so they stay 1).
    let done_init: Vec<u32> = (0..total).map(|id| (id >= internal) as u32).collect();

    let uni = [n as u32, 0, 0, 0];
    let ro = |bytes: &[u8], label| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let rw_init = |bytes: &[u8], label| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("agg uni"),
        contents: bytemuck::cast_slice(&uni),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let lft_buf = ro(bytemuck::cast_slice(left), "agg lft");
    let rgt_buf = ro(bytemuck::cast_slice(right), "agg rgt");
    let mass_buf = rw_init(bytemuck::cast_slice(&mass_init), "agg node mass");
    let com_buf = rw_init(bytemuck::cast_slice(&com_init), "agg node com");
    let lo_buf = rw_init(bytemuck::cast_slice(&lo_init), "agg node lo");
    let hi_buf = rw_init(bytemuck::cast_slice(&hi_init), "agg node hi");
    let done_a = rw_init(bytemuck::cast_slice(&done_init), "agg done a");
    let done_b = rw_init(bytemuck::cast_slice(&done_init), "agg done b");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("aggregate"),
        source: wgpu::ShaderSource::Wgsl(AGGREGATE_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("agg layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, true),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
        ],
    });
    // Two bind groups that swap which "done" buffer is read vs written each pass.
    let make_bind = |din: &wgpu::Buffer, dout: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agg bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uni_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lft_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgt_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mass_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: com_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: din.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dout.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: lo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: hi_buf.as_entire_binding() },
            ],
        })
    };
    let bind_ab = make_bind(&done_a, &done_b);
    let bind_ba = make_bind(&done_b, &done_a);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("agg pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("agg pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch one level per pass until the root (node 0) reports finished. The
    // frontier rises one level each pass, so this takes exactly the tree's height
    // iterations; `total` is a safe never-reached upper bound.
    let groups = (internal as u32).div_ceil(64);
    let mut iter = 0usize;
    loop {
        let (bind, out_buf) = if iter.is_multiple_of(2) {
            (&bind_ab, &done_b)
        } else {
            (&bind_ba, &done_a)
        };
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aggregate level"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Read just the root's flag to decide whether another level is needed.
        let root_back = readback_buffer(device, 4);
        encoder.copy_buffer_to_buffer(out_buf, 0, &root_back, 0, 4);
        queue.submit(Some(encoder.finish()));
        let root_done = map_u32(device, &root_back, 1)[0];
        if root_done == 1 {
            break;
        }
        iter += 1;
        assert!(iter < total, "aggregate refit did not converge");
    }

    // Read the finished node arrays back.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mass_back = readback_buffer(device, (total * 4) as u64);
    let com_back = readback_buffer(device, (total * 16) as u64);
    let lo_back = readback_buffer(device, (total * 16) as u64);
    let hi_back = readback_buffer(device, (total * 16) as u64);
    encoder.copy_buffer_to_buffer(&mass_buf, 0, &mass_back, 0, (total * 4) as u64);
    encoder.copy_buffer_to_buffer(&com_buf, 0, &com_back, 0, (total * 16) as u64);
    encoder.copy_buffer_to_buffer(&lo_buf, 0, &lo_back, 0, (total * 16) as u64);
    encoder.copy_buffer_to_buffer(&hi_buf, 0, &hi_back, 0, (total * 16) as u64);
    queue.submit(Some(encoder.finish()));

    let to_vec4 = |flat: Vec<f32>| -> Vec<[f32; 4]> {
        flat.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    };
    let mass = map_f32(device, &mass_back, total);
    let com = to_vec4(map_f32(device, &com_back, total * 4));
    let lo = to_vec4(map_f32(device, &lo_back, total * 4));
    let hi = to_vec4(map_f32(device, &hi_back, total * 4));
    (mass, com, lo, hi)
}

const TRAVERSE_SHADER: &str = r#"
struct Uni { n: u32, theta2: f32, soft2: f32, g: f32 };
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var<storage, read> node_com: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> node_size2: array<f32>;
@group(0) @binding(3) var<storage, read> lft: array<u32>;
@group(0) @binding(4) var<storage, read> rgt: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_acc: array<vec4<f32>>;

// Barnes–Hut gravity on one particle by walking the tree with a private stack.
// For each node: if it is a far-enough lump (size² < θ²·r²) add its pull and don't
// descend; otherwise push its two children and keep going. Leaves are single
// particles; the leaf that IS this particle contributes nothing because its offset
// r⃗ = 0 makes the numerator m·r⃗ vanish, so no self-force test is needed. Each pull
// uses the softened law a⃗ = G·m·r⃗ / (|r⃗|² + ε²)^{3/2}, matching the CPU octree.
//
// The node record is packed for the walk: `node_com` carries the centre of mass in
// xyz and the total mass in w, and `node_size2` holds the box-diagonal² precomputed
// once (a node is visited by thousands of particles, so computing its size per visit
// was pure waste). So each node touched costs one vec4 read (+ one float if internal),
// not three vec4s plus a size calculation — the walk is memory-bound, so this matters.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= u.n) { return; }
    let leaf_base = u.n - 1u;                 // internal nodes 0…n-2, leaves n-1…2n-2
    let pi = node_com[leaf_base + k].xyz;      // this particle's position

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var stack: array<u32, 64>;
    var sp = 0u;
    stack[0] = 0u; sp = 1u;                    // start at the root (internal node 0)

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = stack[sp];
        let cm = node_com[node];                   // xyz = centre of mass, w = mass
        let d = cm.xyz - pi;
        let r2 = dot(d, d);

        if (node >= leaf_base) {
            // A leaf: one particle. Self gives d=0 → adds nothing.
            let soft = r2 + u.soft2;
            acc = acc + cm.w * d / (soft * sqrt(soft));
        } else {
            // Internal node: lump if far enough, else open it.
            if (node_size2[node] < u.theta2 * r2) {
                let soft = r2 + u.soft2;
                acc = acc + cm.w * d / (soft * sqrt(soft));
            } else if (sp <= 62u) {                  // room for two children
                stack[sp] = lft[node]; sp = sp + 1u;
                stack[sp] = rgt[node]; sp = sp + 1u;
            }
        }
    }
    out_acc[k] = vec4<f32>(acc * u.g, 0.0);
}
"#;

/// Warp-cooperative version of [`TRAVERSE_SHADER`] (same bindings, needs subgroups).
///
/// The particles are Morton-sorted, so the lanes of a subgroup are neighbours in
/// space and want to open nearly the same nodes. Instead of each lane walking its own
/// stack (and the warp stalling on its most-divergent lane), the whole subgroup walks
/// **one** node at a time: a node is opened if `subgroupAny` says *any* lane still
/// needs it, otherwise every lane lumps it. Because that decision is subgroup-uniform,
/// every lane makes identical push/pop moves, so their private stacks stay identical —
/// no shared memory needed, and the control flow never diverges. Each lane still sums
/// the pull for its *own* particle. Out-of-range tail lanes stay in the loop (voting
/// "don't open") so the subgroup control flow remains uniform for `subgroupAny`.
const TRAVERSE_COOP_SHADER: &str = r#"
struct Uni { n: u32, theta2: f32, soft2: f32, g: f32 };
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var<storage, read> node_com: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> node_size2: array<f32>;
@group(0) @binding(3) var<storage, read> lft: array<u32>;
@group(0) @binding(4) var<storage, read> rgt: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_acc: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let live = k < u.n;
    let leaf_base = u.n - 1u;
    // Inactive tail lanes clamp to a valid node so their reads are safe; they never
    // vote to open and never write out, so they cannot affect the result.
    let my_leaf = select(leaf_base, leaf_base + k, live);
    let pi = node_com[my_leaf].xyz;

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var stack: array<u32, 64>;                 // private, but identical across the subgroup
    var sp = 1u;
    stack[0] = 0u;

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let node = stack[sp];                  // uniform across the subgroup
        let cm = node_com[node];
        let d = cm.xyz - pi;
        let r2 = dot(d, d);

        if (node >= leaf_base) {
            let soft = r2 + u.soft2;
            if (live) { acc = acc + cm.w * d / (soft * sqrt(soft)); }
        } else {
            let need_open = live && (node_size2[node] >= u.theta2 * r2);
            if (subgroupAny(need_open)) {      // any lane too close → all open
                if (sp <= 62u) {
                    stack[sp] = lft[node]; sp = sp + 1u;
                    stack[sp] = rgt[node]; sp = sp + 1u;
                }
            } else {                           // all far → all lump
                let soft = r2 + u.soft2;
                if (live) { acc = acc + cm.w * d / (soft * sqrt(soft)); }
            }
        }
    }
    if (live) { out_acc[k] = vec4<f32>(acc * u.g, 0.0); }
}
"#;

/// Barnes–Hut accelerations for every particle, entirely on the GPU.
///
/// What: runs the whole pipeline — bounding box, Morton codes, sort, LBVH build,
/// node mass/COM/box, then a per-particle tree walk — and returns the acceleration
/// on each input particle, in the caller's original order.
/// How/why: this composes the earlier phases. Positions are ordered along the Morton
/// curve (so tree leaves are the sorted particles); after the walk we scatter the
/// per-leaf accelerations back to the original order. The walk itself uses the same
/// opening test and softened law as the CPU [`octree`](super::octree), so it must
/// agree with the direct O(N²) sum to Barnes–Hut accuracy.
/// Principle: Newton's gravity with distant crowds replaced by their centre of mass.
/// Units: `pos` (its `.w` ignored) and `mass` in the caller's units; `theta`
/// dimensionless; `softening` a length; `g` the gravitational constant. Returns
/// accelerations in those units.
pub fn accelerations_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pos: &[[f32; 4]],
    mass: &[f32],
    theta: f32,
    softening: f32,
    g: f32,
) -> Vec<Vec3> {
    let n = pos.len();
    assert!(n >= 2, "need at least two particles");
    assert_eq!(mass.len(), n);

    // Build steps (each reads its input back and re-uploads — fine for a reference
    // path; the real-time version will keep the buffers resident, see the doc).
    let (lo, hi) = bounding_box_gpu(device, queue, pos);
    let inv = Vec3::ONE / (hi - lo).max(Vec3::splat(1e-20)); // guard a flat axis
    let codes = compute_morton_gpu(device, queue, pos, lo, inv);
    let idx: Vec<u32> = (0..n as u32).collect();
    let (sorted_codes, order) = bitonic_sort_gpu(device, queue, &codes, &idx);

    // Reorder particles into the sorted (leaf) order the tree is built on.
    let leaf_com: Vec<[f32; 4]> = order.iter().map(|&o| pos[o as usize]).collect();
    let leaf_mass: Vec<f32> = order.iter().map(|&o| mass[o as usize]).collect();

    let (left, right, parent) = build_lbvh_structure_gpu(device, queue, &sorted_codes);
    let (_node_mass, node_com, node_lo, node_hi) =
        aggregate_nodes_gpu(device, queue, &left, &right, &parent, &leaf_mass, &leaf_com);

    // Precompute each node's size² (box diagonal²) once — the walk reads it directly.
    let node_size2: Vec<f32> = node_lo
        .iter()
        .zip(&node_hi)
        .map(|(lo, hi)| {
            let e = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
        })
        .collect();

    // The tree walk.
    let acc_sorted =
        traverse_gpu(device, queue, &node_com, &node_size2, &left, &right, theta, softening, g);

    // Scatter accelerations back to the caller's original particle order.
    let mut acc = vec![Vec3::ZERO; n];
    for (k, &o) in order.iter().enumerate() {
        acc[o as usize] = acc_sorted[k];
    }
    acc
}

/// Walk the finished tree once per particle and return each one's acceleration.
///
/// What: the [`TRAVERSE_SHADER`] step in isolation — given the node arrays and the
/// child links, returns the acceleration on each leaf particle (in sorted order).
/// How/why: one thread per particle, a private lump-or-open stack walk. Split out so
/// [`accelerations_gpu`] reads cleanly and the walk can be tested on a known tree.
/// Units: as [`accelerations_gpu`].
#[allow(clippy::too_many_arguments)]
fn traverse_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    node_com: &[[f32; 4]],
    node_size2: &[f32],
    left: &[u32],
    right: &[u32],
    theta: f32,
    softening: f32,
    g: f32,
) -> Vec<Vec3> {
    let total = node_com.len();
    let n = total.div_ceil(2); // total = 2n-1
    let uni = Uniforms {
        n: n as u32,
        theta2: theta * theta,
        soft2: softening * softening,
        g,
    };
    let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("traverse uni"),
        contents: bytemuck::bytes_of(&uni),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let ro = |bytes: &[u8], label| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let com_buf = ro(bytemuck::cast_slice(node_com), "traverse com");
    let size_buf = ro(bytemuck::cast_slice(node_size2), "traverse size2");
    let lft_buf = ro(bytemuck::cast_slice(left), "traverse lft");
    let rgt_buf = ro(bytemuck::cast_slice(right), "traverse rgt");
    let acc_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("traverse acc"),
        size: (n * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("traverse"),
        source: wgpu::ShaderSource::Wgsl(TRAVERSE_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("traverse layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
        ],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("traverse bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uni_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: com_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: size_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: lft_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: rgt_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: acc_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("traverse pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("traverse pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("traverse"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    let acc_back = readback_buffer(device, (n * 16) as u64);
    encoder.copy_buffer_to_buffer(&acc_buf, 0, &acc_back, 0, (n * 16) as u64);
    queue.submit(Some(encoder.finish()));

    let flat = map_f32(device, &acc_back, n * 4);
    flat.chunks_exact(4)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect()
}

/// Uniform block for the traversal kernel (16-byte aligned for the GPU).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    theta2: f32,
    soft2: f32,
    g: f32,
}

const BBOX_SHADER: &str = r#"
struct Uni { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: Uni;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_lo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_hi: array<vec4<f32>>;

const WG: u32 = 256u;
const BIG: f32 = 3.4028235e38;   // ~f32::MAX, the identity for min/max

// Scratch shared by the 256 threads of the single workgroup.
var<workgroup> slo: array<vec3<f32>, 256>;
var<workgroup> shi: array<vec3<f32>, 256>;

// Find the axis-aligned box that contains all particles, as a parallel reduction.
// One workgroup: each of the 256 threads first folds its own strided slice of the
// array into a private (min,max), then the threads combine pairwise in shared memory
// (a halving tree) until thread 0 holds the answer. Reduction is correct in any
// order because min/max are associative and commutative.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var lo = vec3<f32>(BIG, BIG, BIG);
    var hi = vec3<f32>(-BIG, -BIG, -BIG);
    var i = t;
    loop {
        if (i >= u.n) { break; }
        let p = pos[i].xyz;
        lo = min(lo, p);
        hi = max(hi, p);
        i = i + WG;                 // grid-stride: thread t handles t, t+256, t+512, …
    }
    slo[t] = lo;
    shi[t] = hi;
    workgroupBarrier();

    // Halving tree: 256→128→…→1 live lanes, each barrier making the writes visible.
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            slo[t] = min(slo[t], slo[t + stride]);
            shi[t] = max(shi[t], shi[t + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        out_lo[0] = vec4<f32>(slo[0], 0.0);
        out_hi[0] = vec4<f32>(shi[0], 0.0);
    }
}
"#;

/// Compute the axis-aligned bounding box of all particle positions on the GPU.
///
/// What: returns `(lo, hi)`, the per-axis minimum and maximum over every particle —
/// the cube the Morton step normalises into.
/// How/why: a parallel reduction in a single workgroup. Each thread first folds a
/// strided slice of the array into a private min/max (so `n` can be far larger than
/// the 256 threads), then the threads combine pairwise through shared memory until
/// thread 0 holds the total. Min and max are associative, so any combination order
/// gives the same result. `workgroupBarrier()` is enough here because everything
/// happens inside one workgroup — no cross-workgroup visibility problem (contrast
/// the level-by-level refit in [`aggregate_nodes_gpu`]).
/// Units: whatever the positions use (AU here); the `.w` lane is ignored.
pub fn bounding_box_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions: &[[f32; 4]],
) -> (Vec3, Vec3) {
    let n = positions.len();
    assert!(n >= 1, "need at least one particle");

    let uni = [n as u32, 0, 0, 0];
    let uni_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bbox uni"),
        contents: bytemuck::cast_slice(&uni),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bbox pos"),
        contents: bytemuck::cast_slice(positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out = |label| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let lo_buf = out("bbox lo");
    let hi_buf = out("bbox hi");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bbox"),
        source: wgpu::ShaderSource::Wgsl(BBOX_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bbox layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bbox bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uni_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: lo_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: hi_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bbox pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bbox pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bbox"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1); // a single workgroup does the whole reduction
    }
    let lo_back = readback_buffer(device, 16);
    let hi_back = readback_buffer(device, 16);
    encoder.copy_buffer_to_buffer(&lo_buf, 0, &lo_back, 0, 16);
    encoder.copy_buffer_to_buffer(&hi_buf, 0, &hi_back, 0, 16);
    queue.submit(Some(encoder.finish()));

    let lo = map_f32(device, &lo_back, 4);
    let hi = map_f32(device, &hi_back, 4);
    (Vec3::new(lo[0], lo[1], lo[2]), Vec3::new(hi[0], hi[1], hi[2]))
}

/// A `MAP_READ` buffer of `bytes` bytes.
fn readback_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Map a readback buffer and copy out `n` `u32`s (blocks until the GPU is done).
fn map_u32(device: &wgpu::Device, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    let data = slice.get_mapped_range();
    bytemuck::cast_slice::<u8, u32>(&data)[..n].to_vec()
}

/// Map a readback buffer and copy out `n` `f32`s (blocks until the GPU is done).
fn map_f32(device: &wgpu::Device, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    let data = slice.get_mapped_range();
    bytemuck::cast_slice::<u8, f32>(&data)[..n].to_vec()
}

/// A uniform-buffer bind-group-layout entry for the compute stage.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A storage-buffer bind-group-layout entry for the compute stage.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Try to acquire a headless GPU device + queue (no window/surface needed).
///
/// What: returns a `(device, queue)` on the best available adapter, or `None` if no
/// GPU is reachable.
/// How/why: a compute-only context — we ask for a high-performance adapter with no
/// `compatible_surface`, so it works off-screen (including in tests). Returning an
/// `Option` lets callers/tests skip gracefully where there is no GPU.
/// Units: none.
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    // Ask for subgroup ops if the adapter has them — the cooperative tree walk uses
    // `subgroupAny`. Falls back cleanly to the scalar walk when absent.
    let want = wgpu::Features::SUBGROUP & adapter.features();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("nbody compute device"),
        required_features: want,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

// ===========================================================================
// Resident stepper: the whole leapfrog step on the GPU, no per-stage readback.
// ===========================================================================
//
// The functions above each build one stage and read it back — perfect for testing,
// wasteful for real time. `GpuNBody` instead keeps *every* buffer resident and runs
// one leapfrog step (kick–drift, rebuild the tree, walk it, kick) as a single command
// submission. The five validated shaders are reused verbatim; a handful of tiny
// "glue" kernels below move data between them on the card (integrate, a Morton that
// reads the box from a buffer, sort setup, gather into sorted order, seed the leaves,
// scatter the forces back). The only thing that ever comes back to the CPU is the
// position buffer, once per frame, to draw.

/// Kick-1 + drift: `v ← v + a·½dt`, then `x ← x + v·dt`. One thread per particle.
const INTEGRATE_PRE_SHADER: &str = r#"
struct U { n: u32, half: f32, dt: f32, _p: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> acc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pos: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    let v = vel[i].xyz + acc[i].xyz * u.half;   // kick with the old acceleration
    let p = pos[i].xyz + v * u.dt;              // drift with the half-kicked velocity
    vel[i] = vec4<f32>(v, 0.0);
    pos[i] = vec4<f32>(p, 0.0);
}
"#;

/// Kick-2: `v ← v + a·½dt` with the freshly computed acceleration.
const INTEGRATE_POST_SHADER: &str = r#"
struct U { n: u32, half: f32, dt: f32, _p: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> acc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    vel[i] = vec4<f32>(vel[i].xyz + acc[i].xyz * u.half, 0.0);
}
"#;

/// Morton codes, reading the bounding box from GPU buffers (see [`MORTON_SHADER`]).
const MORTON_RESIDENT_SHADER: &str = r#"
struct U { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> box_lo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> box_hi: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> codes: array<u32>;
fn expand_bits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    let lo = box_lo[0].xyz;
    let hi = box_hi[0].xyz;
    let inv = 1.0 / max(hi - lo, vec3<f32>(1e-20, 1e-20, 1e-20));
    let q = clamp((pos[i].xyz - lo) * inv, vec3<f32>(0.0), vec3<f32>(1.0));
    let xi = u32(q.x * 1023.0);
    let yi = u32(q.y * 1023.0);
    let zi = u32(q.z * 1023.0);
    codes[i] = (expand_bits(xi) << 2u) | (expand_bits(yi) << 1u) | expand_bits(zi);
}
"#;

/// Fill the padded sort arrays: real `(code, index)` for the first `n`, sentinels
/// after (so they sort to the end, exactly as [`bitonic_sort_gpu`] does on the CPU).
const SORTSETUP_SHADER: &str = r#"
struct U { n: u32, m: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.m) { return; }
    if (i < u.n) {
        keys[i] = codes[i];
        vals[i] = i;
    } else {
        keys[i] = 0xffffffffu;
        vals[i] = 0xffffffffu;
    }
}
"#;

/// Gather particles into Morton (leaf) order: `leaf[k] = particle[ vals[k] ]`.
const GATHER_SHADER: &str = r#"
struct U { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> vals: array<u32>;
@group(0) @binding(4) var<storage, read_write> leaf_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> leaf_mass: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= u.n) { return; }
    let o = vals[k];
    leaf_pos[k] = pos[o];
    leaf_mass[k] = mass[o];
}
"#;

/// Seed the tree for a fresh refit: leaves get their particle's mass/COM/box and are
/// marked finished; internal nodes are marked unfinished. One thread per node.
const AGGSEED_SHADER: &str = r#"
struct U { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> leaf_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> leaf_mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> node_mass: array<f32>;
@group(0) @binding(4) var<storage, read_write> node_com: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> node_lo: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> node_hi: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> done_a: array<u32>;
@group(0) @binding(8) var<storage, read_write> done_b: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = 2u * u.n - 1u;
    if (id >= total) { return; }
    let leaf_base = u.n - 1u;
    if (id >= leaf_base) {
        let k = id - leaf_base;
        let p = leaf_pos[k].xyz;
        node_mass[id] = leaf_mass[k];
        node_com[id] = vec4<f32>(p, leaf_mass[k]);
        node_lo[id] = vec4<f32>(p, 0.0);
        node_hi[id] = vec4<f32>(p, 0.0);
        done_a[id] = 1u;
        done_b[id] = 1u;
    } else {
        done_a[id] = 0u;
        done_b[id] = 0u;
    }
}
"#;

/// Precompute each node's size² (box diagonal²) for the walk. One thread per node.
const SIZE_SHADER: &str = r#"
struct U { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> node_lo: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> node_hi: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> node_size2: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let total = 2u * u.n - 1u;
    if (id >= total) { return; }
    let e = node_hi[id].xyz - node_lo[id].xyz;
    node_size2[id] = dot(e, e);
}
"#;

/// Scatter the per-leaf accelerations back to the original particle order.
const SCATTER_SHADER: &str = r#"
struct U { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> vals: array<u32>;
@group(0) @binding(2) var<storage, read> acc_sorted: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acc: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= u.n) { return; }
    acc[vals[k]] = acc_sorted[k];
}
"#;

/// How many refit levels to run each step. The frontier rises one level per pass, so
/// this must exceed the tree's height; a Morton+index tree over `n` leaves is at most
/// ~`30 + log₂ n` deep, so 64 covers well past the 100k-body target. Extra passes are
/// harmless no-ops (already-finished nodes just carry forward).
const REFIT_PASSES: usize = 64;

/// A fully GPU-resident Barnes–Hut N-body simulation.
///
/// What: owns the particle state (positions, velocities, accelerations, masses) and
/// all the tree scratch in GPU buffers, and steps the whole system with leapfrog on
/// the card — no data crosses to the CPU except the positions, read once per frame to
/// draw.
/// How/why: [`step`](Self::step) encodes the entire kick–drift–force–kick sequence as
/// compute passes into one command buffer and submits it once. Rebuilding the tree
/// every step reuses the validated pipeline (box → codes → sort → LBVH → mass/COM/box
/// → walk); the pass boundaries give the ordering, so there is nothing to synchronise
/// by hand. Buffer sizes are fixed by `n`, so everything is allocated once up front.
/// Units: the caller's (scale-free `G = 1` in the galaxy mode).
pub struct GpuNBody {
    n: usize,
    // State (original particle order).
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    acc: wgpu::Buffer,
    // Uniforms.
    uni_n: wgpu::Buffer,
    uni_sort: wgpu::Buffer,
    uni_trav: wgpu::Buffer,
    uni_integ: wgpu::Buffer,
    bitonic_params: wgpu::Buffer,
    bitonic_passes: usize,
    soft2: f32,
    g: f32,
    // Pipelines.
    p_pre: wgpu::ComputePipeline,
    p_post: wgpu::ComputePipeline,
    p_bbox: wgpu::ComputePipeline,
    p_morton: wgpu::ComputePipeline,
    p_setup: wgpu::ComputePipeline,
    p_bitonic: wgpu::ComputePipeline,
    p_gather: wgpu::ComputePipeline,
    p_seed: wgpu::ComputePipeline,
    p_lbvh: wgpu::ComputePipeline,
    p_agg: wgpu::ComputePipeline,
    p_size: wgpu::ComputePipeline,
    p_trav: wgpu::ComputePipeline,
    p_scatter: wgpu::ComputePipeline,
    // Bind groups.
    b_pre: wgpu::BindGroup,
    b_post: wgpu::BindGroup,
    b_bbox: wgpu::BindGroup,
    b_morton: wgpu::BindGroup,
    b_setup: wgpu::BindGroup,
    b_bitonic: wgpu::BindGroup,
    b_gather: wgpu::BindGroup,
    b_seed: wgpu::BindGroup,
    b_lbvh: wgpu::BindGroup,
    b_agg_ab: wgpu::BindGroup,
    b_agg_ba: wgpu::BindGroup,
    b_size: wgpu::BindGroup,
    b_trav: wgpu::BindGroup,
    b_scatter: wgpu::BindGroup,
    // Readback for drawing.
    pos_readback: wgpu::Buffer,
}

impl GpuNBody {
    /// Build the resident simulation and compute the starting accelerations.
    ///
    /// What: uploads the particles, allocates every scratch buffer, wires up all the
    /// pipelines, and primes `acc` so the first [`step`](Self::step) is a valid kick.
    /// How/why: sizes come straight from `n` (leaves `n`, nodes `2n-1`, sort padded to
    /// the next power of two), so this is a one-time setup; the per-step work then
    /// allocates nothing.
    /// Units: as the struct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: &[Vec3],
        vel: &[Vec3],
        mass: &[f32],
        theta: f32,
        softening: f32,
        g: f32,
    ) -> Self {
        let n = pos.len();
        assert!(n >= 2, "need at least two particles");
        assert_eq!(vel.len(), n);
        assert_eq!(mass.len(), n);
        let total = 2 * n - 1;
        let internal = n - 1;
        let m = n.next_power_of_two();

        let pos4: Vec<[f32; 4]> = pos.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let vel4: Vec<[f32; 4]> = vel.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();

        // --- state buffers ---
        let storage = wgpu::BufferUsages::STORAGE;
        let sc = storage | wgpu::BufferUsages::COPY_SRC;
        let init = |data: &[u8], usage, label| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
        };
        let zeros = |bytes: u64, usage, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let pos_buf = init(bytemuck::cast_slice(&pos4), sc | wgpu::BufferUsages::COPY_DST, "pos");
        let vel_buf = init(bytemuck::cast_slice(&vel4), storage, "vel");
        let acc_buf = zeros((n * 16) as u64, storage, "acc");
        let mass_buf = init(bytemuck::cast_slice(mass), storage, "mass");

        // --- scratch buffers ---
        let box_lo = zeros(16, storage, "box lo");
        let box_hi = zeros(16, storage, "box hi");
        let codes = zeros((n * 4) as u64, storage, "codes");
        let keys = zeros((m * 4) as u64, storage, "keys");
        let vals = zeros((m * 4) as u64, storage, "vals");
        let leaf_pos = zeros((n * 16) as u64, storage, "leaf pos");
        let leaf_mass = zeros((n * 4) as u64, storage, "leaf mass");
        let lft = zeros((internal * 4) as u64, storage, "lft");
        let rgt = zeros((internal * 4) as u64, storage, "rgt");
        let par = zeros((total * 4) as u64, storage, "par"); // written by LBVH, unused after
        let node_mass = zeros((total * 4) as u64, storage, "node mass");
        let node_com = zeros((total * 16) as u64, storage, "node com");
        let node_lo = zeros((total * 16) as u64, storage, "node lo");
        let node_hi = zeros((total * 16) as u64, storage, "node hi");
        let node_size2 = zeros((total * 4) as u64, storage, "node size2");
        let done_a = zeros((total * 4) as u64, storage, "done a");
        let done_b = zeros((total * 4) as u64, storage, "done b");
        let acc_sorted = zeros((n * 16) as u64, storage, "acc sorted");
        let pos_readback = readback_buffer(device, (n * 16) as u64);

        // --- uniforms ---
        let uni_n = init(bytemuck::cast_slice(&[n as u32, 0, 0, 0]), wgpu::BufferUsages::UNIFORM, "uni n");
        let uni_sort = init(
            bytemuck::cast_slice(&[n as u32, m as u32, 0, 0]),
            wgpu::BufferUsages::UNIFORM,
            "uni sort",
        );
        let trav = Uniforms { n: n as u32, theta2: theta * theta, soft2: softening * softening, g };
        let uni_trav = init(bytemuck::bytes_of(&trav), wgpu::BufferUsages::UNIFORM, "uni trav");
        let uni_integ = init(
            bytemuck::cast_slice(&[f32::from_bits(n as u32), 0.0f32, 0.0f32, 0.0f32]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "uni integ",
        );

        // Bitonic schedule packed as 256-byte-aligned uniform blocks with a dynamic
        // offset per sub-pass (identical to `bitonic_sort_gpu`).
        const STRIDE: usize = 256;
        let mut schedule: Vec<(u32, u32)> = Vec::new();
        let mut kk = 2usize;
        while kk <= m {
            let mut j = kk / 2;
            while j >= 1 {
                schedule.push((j as u32, kk as u32));
                j /= 2;
            }
            kk *= 2;
        }
        let mut params = vec![0u8; schedule.len().max(1) * STRIDE];
        for (idx, &(j, k)) in schedule.iter().enumerate() {
            let base = idx * STRIDE;
            params[base..base + 4].copy_from_slice(&j.to_le_bytes());
            params[base + 4..base + 8].copy_from_slice(&k.to_le_bytes());
            params[base + 8..base + 12].copy_from_slice(&(m as u32).to_le_bytes());
        }
        let bitonic_params =
            init(&params, wgpu::BufferUsages::UNIFORM, "bitonic params");

        // --- pipelines + bind groups ---
        let mk = |src, entries: &[wgpu::BindGroupLayoutEntry], label: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipe, layout)
        };
        let bind = |layout: &wgpu::BindGroupLayout, res: &[&wgpu::Buffer], label: &str| {
            let entries: Vec<wgpu::BindGroupEntry> = res
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };

        let (p_pre, l_pre) = mk(
            INTEGRATE_PRE_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false), storage_entry(3, false)],
            "pre",
        );
        let b_pre = bind(&l_pre, &[&uni_integ, &acc_buf, &vel_buf, &pos_buf], "pre");

        let (p_post, l_post) = mk(
            INTEGRATE_POST_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
            "post",
        );
        let b_post = bind(&l_post, &[&uni_integ, &acc_buf, &vel_buf], "post");

        let (p_bbox, l_bbox) = mk(
            BBOX_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false), storage_entry(3, false)],
            "bbox",
        );
        let b_bbox = bind(&l_bbox, &[&uni_n, &pos_buf, &box_lo, &box_hi], "bbox");

        let (p_morton, l_morton) = mk(
            MORTON_RESIDENT_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
            "morton",
        );
        let b_morton = bind(&l_morton, &[&uni_n, &pos_buf, &box_lo, &box_hi, &codes], "morton");

        let (p_setup, l_setup) = mk(
            SORTSETUP_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false), storage_entry(3, false)],
            "setup",
        );
        let b_setup = bind(&l_setup, &[&uni_sort, &codes, &keys, &vals], "setup");

        // Bitonic reuses the shared shader but with a dynamic-offset uniform.
        let (p_bitonic, b_bitonic) = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bitonic"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BITONIC_SHADER)),
            });
            let mut ulayout = uniform_entry(0);
            ulayout.ty = wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(16),
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bitonic"),
                entries: &[ulayout, storage_entry(1, false), storage_entry(2, false)],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bitonic"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bitonic"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bitonic"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &bitonic_params,
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vals.as_entire_binding() },
                ],
            });
            (pipe, bg)
        };

        let (p_gather, l_gather) = mk(
            GATHER_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
            "gather",
        );
        let b_gather = bind(
            &l_gather,
            &[&uni_n, &pos_buf, &mass_buf, &vals, &leaf_pos, &leaf_mass],
            "gather",
        );

        let (p_seed, l_seed) = mk(
            AGGSEED_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
            "seed",
        );
        let b_seed = bind(
            &l_seed,
            &[
                &uni_n, &leaf_pos, &leaf_mass, &node_mass, &node_com, &node_lo, &node_hi, &done_a,
                &done_b,
            ],
            "seed",
        );

        let (p_lbvh, l_lbvh) = mk(
            LBVH_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
            "lbvh",
        );
        let b_lbvh = bind(&l_lbvh, &[&uni_n, &keys, &lft, &rgt, &par], "lbvh");

        let (p_agg, l_agg) = mk(
            AGGREGATE_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
            "agg",
        );
        let agg_bind = |din: &wgpu::Buffer, dout: &wgpu::Buffer| {
            bind(
                &l_agg,
                &[&uni_n, &lft, &rgt, &node_mass, &node_com, din, dout, &node_lo, &node_hi],
                "agg",
            )
        };
        let b_agg_ab = agg_bind(&done_a, &done_b);
        let b_agg_ba = agg_bind(&done_b, &done_a);

        let (p_size, l_size) = mk(
            SIZE_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
            "size",
        );
        let b_size = bind(&l_size, &[&uni_n, &node_lo, &node_hi, &node_size2], "size");

        // Use the warp-cooperative walk where subgroups are available (same bindings).
        let trav_src = if device.features().contains(wgpu::Features::SUBGROUP) {
            TRAVERSE_COOP_SHADER
        } else {
            TRAVERSE_SHADER
        };
        let (p_trav, l_trav) = mk(
            trav_src,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
            "trav",
        );
        let b_trav = bind(
            &l_trav,
            &[&uni_trav, &node_com, &node_size2, &lft, &rgt, &acc_sorted],
            "trav",
        );

        let (p_scatter, l_scatter) = mk(
            SCATTER_SHADER,
            &[uniform_entry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false)],
            "scatter",
        );
        let b_scatter = bind(&l_scatter, &[&uni_n, &vals, &acc_sorted, &acc_buf], "scatter");

        let sim = GpuNBody {
            n,
            pos: pos_buf,
            vel: vel_buf,
            acc: acc_buf,
            uni_n,
            uni_sort,
            uni_trav,
            uni_integ,
            bitonic_params,
            bitonic_passes: schedule.len(),
            soft2: softening * softening,
            g,
            p_pre,
            p_post,
            p_bbox,
            p_morton,
            p_setup,
            p_bitonic,
            p_gather,
            p_seed,
            p_lbvh,
            p_agg,
            p_size,
            p_trav,
            p_scatter,
            b_pre,
            b_post,
            b_bbox,
            b_morton,
            b_setup,
            b_bitonic,
            b_gather,
            b_seed,
            b_lbvh,
            b_agg_ab,
            b_agg_ba,
            b_size,
            b_trav,
            b_scatter,
            pos_readback,
        };
        // Prime the accelerations so the first kick is valid.
        sim.recompute_forces(device, queue);
        sim
    }

    /// Encode the force calculation (tree rebuild + walk) into an existing encoder.
    ///
    /// What: appends every stage from bounding box to the scatter that writes the
    /// fresh accelerations into `acc`.
    /// How/why: each stage is its own compute pass, so wgpu inserts the barriers that
    /// make one stage's writes visible to the next; the refit is run a fixed
    /// [`REFIT_PASSES`] times (no readback needed to know when it converged).
    /// Units: none.
    fn encode_forces(&self, enc: &mut wgpu::CommandEncoder) {
        let gn = (self.n as u32).div_ceil(64);
        let gtot = ((2 * self.n - 1) as u32).div_ceil(64);
        let gm = (self.n.next_power_of_two() as u32).div_ceil(64);
        // A single dispatch in its own compute pass (the pass boundary is the barrier).
        fn one(
            enc: &mut wgpu::CommandEncoder,
            pipe: &wgpu::ComputePipeline,
            bg: &wgpu::BindGroup,
            offset: &[u32],
            groups: u32,
        ) {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            p.set_pipeline(pipe);
            p.set_bind_group(0, bg, offset);
            p.dispatch_workgroups(groups, 1, 1);
        }
        one(enc, &self.p_bbox, &self.b_bbox, &[], 1); // single-workgroup reduction
        one(enc, &self.p_morton, &self.b_morton, &[], gn);
        one(enc, &self.p_setup, &self.b_setup, &[], gm);
        // Bitonic: one pass per sub-pass, selecting its (j,k) by dynamic offset.
        for idx in 0..self.bitonic_passes {
            one(enc, &self.p_bitonic, &self.b_bitonic, &[(idx * 256) as u32], gm);
        }
        one(enc, &self.p_gather, &self.b_gather, &[], gn);
        one(enc, &self.p_seed, &self.b_seed, &[], gtot);
        one(enc, &self.p_lbvh, &self.b_lbvh, &[], gn);
        for i in 0..REFIT_PASSES {
            let bg = if i.is_multiple_of(2) { &self.b_agg_ab } else { &self.b_agg_ba };
            one(enc, &self.p_agg, bg, &[], gtot);
        }
        one(enc, &self.p_size, &self.b_size, &[], gtot); // pack node size² for the walk
        one(enc, &self.p_trav, &self.b_trav, &[], gn);
        one(enc, &self.p_scatter, &self.b_scatter, &[], gn);
    }

    /// Compute accelerations for the current positions (used once at start-up).
    fn recompute_forces(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("forces") });
        self.encode_forces(&mut enc);
        queue.submit(Some(enc.finish()));
    }

    /// Advance the whole system by one leapfrog step of size `dt`, on the GPU.
    ///
    /// What: kick-drift with the stored acceleration, rebuild the tree and recompute
    /// the acceleration, then the second kick — all in one command submission.
    /// How/why: `acc` already holds the acceleration at the current positions (from
    /// start-up or the previous step), so the split kick is valid; nothing is read
    /// back, so the CPU never waits mid-step.
    /// Units: `dt` in the caller's time unit.
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        // Refresh the integrator uniform (n reinterpreted as a float bit-pattern in
        // the shader's u32 slot; half and dt are real floats).
        let half = 0.5 * dt;
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&(self.n as u32).to_le_bytes());
        bytes[4..8].copy_from_slice(&half.to_le_bytes());
        bytes[8..12].copy_from_slice(&dt.to_le_bytes());
        queue.write_buffer(&self.uni_integ, 0, &bytes);

        let gn = (self.n as u32).div_ceil(64);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("step") });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kick+drift"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.p_pre);
            p.set_bind_group(0, &self.b_pre, &[]);
            p.dispatch_workgroups(gn, 1, 1);
        }
        self.encode_forces(&mut enc);
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kick"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.p_post);
            p.set_bind_group(0, &self.b_post, &[]);
            p.dispatch_workgroups(gn, 1, 1);
        }
        queue.submit(Some(enc.finish()));
    }

    /// Change the Barnes–Hut opening angle θ live (bigger = faster, a bit rougher).
    ///
    /// What: rewrites the traversal uniform with the new `θ²`, so the next step lumps
    /// nodes more (larger θ) or less (smaller θ) aggressively.
    /// How/why: only the walk's accept test `size² < θ²·r²` uses θ, so a single small
    /// uniform write is all it takes — no rebuild.
    /// Units: `theta` dimensionless.
    pub fn set_theta(&self, queue: &wgpu::Queue, theta: f32) {
        let uni = Uniforms {
            n: self.n as u32,
            theta2: theta * theta,
            soft2: self.soft2,
            g: self.g,
        };
        queue.write_buffer(&self.uni_trav, 0, bytemuck::bytes_of(&uni));
    }

    /// Copy the positions back to the CPU (once per frame, to draw).
    ///
    /// What: returns the current position of every particle, in the original order.
    /// How/why: the only place data leaves the GPU — one copy of the position buffer
    /// into a mapped staging buffer. Everything else stays resident.
    /// Units: the caller's length.
    pub fn positions(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec3> {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(&self.pos, 0, &self.pos_readback, 0, (self.n * 16) as u64);
        queue.submit(Some(enc.finish()));
        let flat = map_f32(device, &self.pos_readback, self.n * 4);
        // The staging buffer is reused every frame, so it must be unmapped again
        // before the next copy/submit — otherwise wgpu rejects the submit.
        self.pos_readback.unmap();
        flat.chunks_exact(4).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
    }

    /// The resident position buffer, for the renderer to draw from directly.
    ///
    /// What: the GPU buffer holding every particle's position (`vec4`, xyz used).
    /// How/why: it is `STORAGE`, so a vertex shader can read it by instance index —
    /// no copy to the CPU. Positions update in place each [`step`](Self::step).
    /// Units: the caller's length.
    pub fn pos_buffer(&self) -> &wgpu::Buffer {
        &self.pos
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the system is empty (never, but clippy likes the pair).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Time each stage of a step in isolation (test/diagnostic only).
    ///
    /// Returns `(stage_name, milliseconds)` averaged over `iters` runs, submitting and
    /// waiting on each stage separately so we can see where the time goes.
    #[cfg(test)]
    pub(crate) fn profile(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        iters: usize,
    ) -> Vec<(&'static str, f64)> {
        use std::time::Instant;
        fn one(
            enc: &mut wgpu::CommandEncoder,
            pipe: &wgpu::ComputePipeline,
            bg: &wgpu::BindGroup,
            offset: &[u32],
            groups: u32,
        ) {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            p.set_pipeline(pipe);
            p.set_bind_group(0, bg, offset);
            p.dispatch_workgroups(groups, 1, 1);
        }
        let gn = (self.n as u32).div_ceil(64);
        let gtot = ((2 * self.n - 1) as u32).div_ceil(64);
        let gm = (self.n.next_power_of_two() as u32).div_ceil(64);

        let bench = |label: &'static str, f: &mut dyn FnMut(&mut wgpu::CommandEncoder)| {
            for _ in 0..3 {
                let mut e = device.create_command_encoder(&Default::default());
                f(&mut e);
                queue.submit(Some(e.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut e = device.create_command_encoder(&Default::default());
                f(&mut e);
                queue.submit(Some(e.finish()));
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            (label, t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
        };

        let mut out = Vec::new();
        out.push(bench("bbox", &mut |e| one(e, &self.p_bbox, &self.b_bbox, &[], 1)));
        out.push(bench("morton", &mut |e| one(e, &self.p_morton, &self.b_morton, &[], gn)));
        out.push(bench("sort-setup", &mut |e| one(e, &self.p_setup, &self.b_setup, &[], gm)));
        out.push(bench("bitonic-sort", &mut |e| {
            for idx in 0..self.bitonic_passes {
                one(e, &self.p_bitonic, &self.b_bitonic, &[(idx * 256) as u32], gm);
            }
        }));
        out.push(bench("gather", &mut |e| one(e, &self.p_gather, &self.b_gather, &[], gn)));
        out.push(bench("seed", &mut |e| one(e, &self.p_seed, &self.b_seed, &[], gtot)));
        out.push(bench("lbvh-build", &mut |e| one(e, &self.p_lbvh, &self.b_lbvh, &[], gn)));
        out.push(bench("refit(64)", &mut |e| {
            for i in 0..REFIT_PASSES {
                let bg = if i.is_multiple_of(2) { &self.b_agg_ab } else { &self.b_agg_ba };
                one(e, &self.p_agg, bg, &[], gtot);
            }
        }));
        out.push(bench("traverse", &mut |e| one(e, &self.p_trav, &self.b_trav, &[], gn)));
        out.push(bench("scatter", &mut |e| one(e, &self.p_scatter, &self.b_scatter, &[], gn)));
        out.push(bench("full-forces", &mut |e| self.encode_forces(e)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The GPU Morton codes must match the CPU reference bit-for-bit.
    ///
    /// Both quantise the same way, so any difference would mean a real bug in the
    /// kernel (or a mismatch with the CPU order the tree will be built on).
    #[test]
    fn gpu_morton_matches_cpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
        let n = 4000;
        let pos: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.unit() as f32 * 6.0 - 3.0,
                    rng.unit() as f32 * 4.0 - 2.0,
                    rng.unit() as f32 * 5.0 - 2.5,
                )
            })
            .collect();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let inv = Vec3::ONE / (hi - lo);

        let cpu: Vec<u32> = pos.iter().map(|p| morton_code(*p, lo, inv)).collect();
        let packed: Vec<[f32; 4]> = pos.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let gpu = compute_morton_gpu(&device, &queue, &packed, lo, inv);

        assert_eq!(cpu.len(), gpu.len());
        assert_eq!(cpu, gpu, "GPU Morton codes differ from the CPU reference");
    }

    /// The GPU bitonic sort must match a plain CPU sort by `(key, index)`.
    ///
    /// Keys are drawn from a small range to force many ties, so this also checks
    /// that equal keys end up ordered by index (what the tree build relies on).
    #[test]
    fn gpu_sort_matches_cpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0xFEED_FACE_C0DE_1234);
        let n = 5000; // not a power of two, to exercise padding
        let keys: Vec<u32> = (0..n).map(|_| (rng.unit() * 1000.0) as u32).collect();
        let vals: Vec<u32> = (0..n as u32).collect();

        let mut pairs: Vec<(u32, u32)> = keys.iter().copied().zip(vals.iter().copied()).collect();
        pairs.sort(); // lexicographic (key, then index)
        let ck: Vec<u32> = pairs.iter().map(|p| p.0).collect();
        let cv: Vec<u32> = pairs.iter().map(|p| p.1).collect();

        let (gk, gv) = bitonic_sort_gpu(&device, &queue, &keys, &vals);
        assert_eq!(ck, gk, "sorted keys differ from CPU");
        assert_eq!(cv, gv, "sorted order (payload) differs from CPU");
    }

    /// The GPU LBVH must be a correct binary radix tree over the sorted codes.
    ///
    /// Codes are drawn from a small range so there are many duplicates — which
    /// forces the δ index-tiebreak path in the Karras build. We then check, on the
    /// read-back structure, that a left-to-right walk from the root visits leaves
    /// 0,1,…,N-1 in order (a valid tree covering every particle exactly once) and
    /// that every child points back to its parent.
    #[test]
    fn gpu_lbvh_is_a_valid_tree() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x0BAD_F00D_1234_5678);
        let n = 3000usize;
        let mut codes: Vec<u32> = (0..n).map(|_| (rng.unit() * 2000.0) as u32).collect();
        codes.sort(); // sorted, with many ties

        let (left, right, parent) = build_lbvh_structure_gpu(&device, &queue, &codes);
        assert_eq!(left.len(), n - 1);
        assert_eq!(right.len(), n - 1);
        assert_eq!(parent.len(), 2 * n - 1);

        // Root's parent is the sentinel; children point back at their parents.
        assert_eq!(parent[0], NO_NODE, "root should have no parent");
        for i in 0..(n - 1) {
            assert_eq!(parent[left[i] as usize], i as u32, "left child of {i}");
            assert_eq!(parent[right[i] as usize], i as u32, "right child of {i}");
        }

        // Left-to-right leaf walk from the root must be 0,1,…,N-1.
        let leaf_base = (n - 1) as u32;
        let mut leaves = Vec::with_capacity(n);
        let mut stack = vec![0u32];
        let mut guard = 0;
        while let Some(node) = stack.pop() {
            guard += 1;
            assert!(guard <= 4 * n, "traversal did not terminate (malformed tree)");
            if node >= leaf_base {
                leaves.push(node - leaf_base);
            } else {
                stack.push(right[node as usize]);
                stack.push(left[node as usize]);
            }
        }
        let expected: Vec<u32> = (0..n as u32).collect();
        assert_eq!(leaves, expected, "leaves not visited in sorted order");
    }

    /// The GPU bounding-box reduction must match a plain CPU min/max over the points.
    ///
    /// `n` is deliberately not a multiple of the 256-thread workgroup, so the
    /// grid-stride tail (threads that fold a different number of elements) is
    /// exercised. Min/max are exact integers of the float inputs, so we compare for
    /// exact equality.
    #[test]
    fn gpu_bounding_box_matches_cpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x00C0_FFEE_1234_5678);
        let n = 12345; // not a multiple of 256
        let pos: Vec<[f32; 4]> = (0..n)
            .map(|_| {
                [
                    rng.unit() as f32 * 20.0 - 7.0,
                    rng.unit() as f32 * 9.0 - 5.0,
                    rng.unit() as f32 * 13.0 - 2.0,
                    0.0,
                ]
            })
            .collect();

        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &pos {
            let v = Vec3::new(p[0], p[1], p[2]);
            lo = lo.min(v);
            hi = hi.max(v);
        }

        let (glo, ghi) = bounding_box_gpu(&device, &queue, &pos);
        assert_eq!(glo, lo, "GPU lo differs from CPU");
        assert_eq!(ghi, hi, "GPU hi differs from CPU");
    }

    /// The GPU bottom-up aggregate (mass + centre of mass) must match a CPU refit of
    /// the same tree, and the root must hold the totals over all particles.
    ///
    /// We build a real tree from random points, hand the GPU random leaf masses, and
    /// recompute every node on the CPU by walking the read-back structure in
    /// reverse-preorder (children before parents). The atomic walk-up is the only
    /// place we rely on GPU atomics for ordering, so this checks it end to end.
    #[test]
    fn gpu_node_aggregates_match_cpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0xA5A5_1234_DEAD_BEEF);
        let n = 4000usize;

        // Random points, their bounding box, and Morton codes — a genuine tree.
        let pos: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.unit() as f32 * 6.0 - 3.0,
                    rng.unit() as f32 * 4.0 - 2.0,
                    rng.unit() as f32 * 5.0 - 2.5,
                )
            })
            .collect();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let inv = Vec3::ONE / (hi - lo);

        // Sort (code, index) so the leaf order matches what the tree is built on.
        let mut keyed: Vec<(u32, usize)> = pos
            .iter()
            .enumerate()
            .map(|(i, p)| (morton_code(*p, lo, inv), i))
            .collect();
        keyed.sort();
        let codes: Vec<u32> = keyed.iter().map(|k| k.0).collect();
        let leaf_mass: Vec<f32> = keyed.iter().map(|_| 0.5 + rng.unit() as f32).collect();
        let leaf_com: Vec<[f32; 4]> = keyed
            .iter()
            .map(|k| {
                let p = pos[k.1];
                [p.x, p.y, p.z, 0.0]
            })
            .collect();

        let (left, right, parent) = build_lbvh_structure_gpu(&device, &queue, &codes);
        let (mass, com, node_lo, node_hi) =
            aggregate_nodes_gpu(&device, &queue, &left, &right, &parent, &leaf_mass, &leaf_com);

        let total = 2 * n - 1;
        assert_eq!(mass.len(), total);
        assert_eq!(com.len(), total);
        assert_eq!(node_lo.len(), total);
        assert_eq!(node_hi.len(), total);

        // CPU refit: preorder from the root, then process in reverse so every node's
        // two children are finished before the node itself.
        let leaf_base = (n - 1) as u32;
        let mut order = Vec::with_capacity(total);
        let mut stack = vec![0u32];
        while let Some(node) = stack.pop() {
            order.push(node);
            if node < leaf_base {
                stack.push(right[node as usize]);
                stack.push(left[node as usize]);
            }
        }
        let mut cmass = vec![0.0f32; total];
        let mut ccom = vec![[0.0f32; 3]; total];
        let mut clo = vec![[0.0f32; 3]; total];
        let mut chi = vec![[0.0f32; 3]; total];
        for &node in order.iter().rev() {
            let id = node as usize;
            if node >= leaf_base {
                let k = (node - leaf_base) as usize;
                let p = [leaf_com[k][0], leaf_com[k][1], leaf_com[k][2]];
                cmass[id] = leaf_mass[k];
                ccom[id] = p;
                clo[id] = p;
                chi[id] = p;
            } else {
                let (l, r) = (left[id] as usize, right[id] as usize);
                let (ml, mr) = (cmass[l], cmass[r]);
                let m = ml + mr;
                cmass[id] = m;
                for a in 0..3 {
                    ccom[id][a] = (ml * ccom[l][a] + mr * ccom[r][a]) / m;
                    clo[id][a] = clo[l][a].min(clo[r][a]);
                    chi[id][a] = chi[l][a].max(chi[r][a]);
                }
            }
        }

        // Root totals: mass = Σ leaf masses, COM = mass-weighted mean of the points.
        let total_mass: f32 = leaf_mass.iter().sum();
        assert!(
            (mass[0] - total_mass).abs() <= 1e-2 * total_mass,
            "root mass {} vs {}",
            mass[0],
            total_mass
        );

        // Every node must match the CPU refit closely (same tree, same formula).
        for id in 0..total {
            assert!(
                (mass[id] - cmass[id]).abs() <= 1e-3 * cmass[id].max(1.0),
                "mass mismatch at node {id}: {} vs {}",
                mass[id],
                cmass[id]
            );
            for a in 0..3 {
                assert!(
                    (com[id][a] - ccom[id][a]).abs() <= 2e-3,
                    "com mismatch at node {id} axis {a}: {} vs {}",
                    com[id][a],
                    ccom[id][a]
                );
                // The box (min/max) is built from exact copies, so it must match
                // exactly — this is what the traversal uses for each node's size.
                assert_eq!(node_lo[id][a], clo[id][a], "lo mismatch at node {id} axis {a}");
                assert_eq!(node_hi[id][a], chi[id][a], "hi mismatch at node {id} axis {a}");
            }
        }
    }

    /// The full GPU Barnes–Hut pipeline must agree with the exact O(N²) sum.
    ///
    /// This is the same yardstick the CPU octree uses: build a random cloud, compute
    /// every particle's acceleration on the GPU (bbox → Morton → sort → tree → walk),
    /// and compare against the direct softened sum. The GPU tree is a *binary* LBVH,
    /// not the CPU's octree, so the two trees differ node-for-node — hence we check
    /// against the ground-truth direct sum, not the CPU tree, and only to Barnes–Hut
    /// accuracy (θ = 0.5): small mean error, bounded worst case.
    #[test]
    fn gpu_accelerations_match_direct_sum() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x1357_9BDF_2468_ACE0);
        let n = 3000usize;
        let soft = 0.05f32;
        let theta = 0.5f32;
        let g = 1.0f32;

        let pos3: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(
                    rng.unit() as f32 * 2.0 - 1.0,
                    rng.unit() as f32 * 2.0 - 1.0,
                    rng.unit() as f32 * 2.0 - 1.0,
                )
            })
            .collect();
        let mass: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit() as f32).collect();
        let pos4: Vec<[f32; 4]> = pos3.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();

        let gpu = accelerations_gpu(&device, &queue, &pos4, &mass, theta, soft, g);
        assert_eq!(gpu.len(), n);

        // Direct softened O(N²) sum — the ground truth.
        let soft2 = soft * soft;
        let direct = |i: usize| -> Vec3 {
            let mut a = Vec3::ZERO;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = pos3[j] - pos3[i];
                let s = d.length_squared() + soft2;
                a += mass[j] * d / (s * s.sqrt());
            }
            a * g
        };

        let mut sum_rel = 0.0f32;
        let mut worst = 0.0f32;
        for i in 0..n {
            let d = direct(i);
            let rel = (gpu[i] - d).length() / d.length().max(1e-6);
            sum_rel += rel;
            worst = worst.max(rel);
        }
        let mean = sum_rel / n as f32;
        assert!(mean < 0.01, "mean relative error {mean} too high");
        assert!(worst < 0.15, "worst relative error {worst} too high");
    }

    /// The resident GPU leapfrog must track the CPU leapfrog step for step.
    ///
    /// Both integrate the *same* initial conditions with exact forces (θ = 0, so the
    /// GPU tree opens down to every leaf and matches the direct sum), so after many
    /// steps their particles should still sit almost on top of each other — only f32
    /// summation-order rounding separates them. This exercises the whole resident
    /// pipeline: kick–drift, tree rebuild, gather into Morton order, walk, scatter
    /// back, second kick — any mistake in that plumbing shows up as drift.
    #[test]
    fn resident_stepper_matches_cpu_leapfrog() {
        use crate::physics::particles::Particles;

        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x2468_0ACE_1357_9BDF);
        let n = 400usize;
        let soft = 0.1f32;
        let g = 1.0f32;
        let dt = 0.005f32;

        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        for _ in 0..n {
            pos.push(Vec3::new(
                rng.unit() as f32 * 2.0 - 1.0,
                rng.unit() as f32 * 2.0 - 1.0,
                rng.unit() as f32 * 2.0 - 1.0,
            ));
            vel.push(Vec3::new(
                rng.unit() as f32 - 0.5,
                rng.unit() as f32 - 0.5,
                rng.unit() as f32 - 0.5,
            ) * 0.1);
        }
        let mass = vec![1.0f32; n];

        let mut cpu = Particles::new(pos.clone(), vel.clone(), mass.clone(), 0.0, soft, g);
        let gpu = GpuNBody::new(&device, &queue, &pos, &vel, &mass, 0.0, soft, g);

        let steps = 50;
        for _ in 0..steps {
            cpu.step(dt);
            gpu.step(&device, &queue, dt);
        }
        let gpos = gpu.positions(&device, &queue);

        let mut sum = 0.0f32;
        let mut worst = 0.0f32;
        for i in 0..n {
            let d = (gpos[i] - cpu.pos[i]).length();
            sum += d;
            worst = worst.max(d);
        }
        let mean = sum / n as f32;
        // The cluster spans ~2 units; exact-force leapfrogs must stay within a whisker.
        assert!(mean < 0.02, "mean position drift {mean} too high");
        assert!(worst < 0.1, "worst position drift {worst} too high");
    }

    /// Per-stage timing on the real GPU at galaxy scale. Ignored by default; run with
    /// `cargo test --release profile_resident_stages -- --ignored --nocapture`.
    ///
    /// Each stage is submitted on its own and waited on, so the printed milliseconds
    /// over-serialise a little versus the fused step, but the *relative* costs show
    /// exactly where a step spends its time — which is what tells us what to optimise.
    #[test]
    #[ignore]
    fn profile_resident_stages() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0xB16_5CA1E_0F16_1234);
        // Override with SOLSIM_N to profile other sizes (e.g. a million).
        let n = std::env::var("SOLSIM_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_002usize);
        println!("profiling n = {n}");
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        for _ in 0..n {
            pos.push(Vec3::new(
                rng.unit() as f32 * 4.0 - 2.0,
                rng.unit() as f32 * 4.0 - 2.0,
                rng.unit() as f32 * 4.0 - 2.0,
            ));
            vel.push(Vec3::ZERO);
        }
        let mass = vec![4.0f32 / n as f32; n];
        let theta = std::env::var("SOLSIM_THETA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.6f32);
        let sim = GpuNBody::new(&device, &queue, &pos, &vel, &mass, theta, 0.05, 1.0);
        for row in sim.profile(&device, &queue, 30) {
            println!("{:>14}  {:6.3} ms", row.0, row.1);
        }
    }

    /// At galaxy scale the resident stepper must stay finite and bounded.
    ///
    /// This exercises the real code paths the 60k-body galaxy uses: a large bitonic
    /// sort, the fixed 64-pass refit against a deep tree, and the 64-entry traversal
    /// stack. `n` is a power of two, so the sort needs no padding — a worthwhile edge
    /// case. We step a self-gravitating cloud and check nothing blows up (no NaNs, and
    /// the cloud does not explode), which would flag an out-of-bounds tree or stack.
    #[test]
    fn resident_stepper_large_stays_finite() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
        let n = 16384usize; // power of two: sort runs with no sentinel padding
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        for _ in 0..n {
            pos.push(Vec3::new(
                rng.unit() as f32 * 4.0 - 2.0,
                rng.unit() as f32 * 4.0 - 2.0,
                rng.unit() as f32 * 4.0 - 2.0,
            ));
            vel.push(Vec3::ZERO);
        }
        let mass = vec![1.0f32 / n as f32; n]; // light particles: gentle, bounded motion

        let gpu = GpuNBody::new(&device, &queue, &pos, &vel, &mass, 0.6, 0.1, 1.0);
        // Step *and read back* every iteration, exactly like the frame loop — this is
        // what catches the reused staging buffer being left mapped between frames.
        let mut out = Vec::new();
        for _ in 0..6 {
            gpu.step(&device, &queue, 0.01);
            out = gpu.positions(&device, &queue);
        }
        assert_eq!(out.len(), n);
        for (i, p) in out.iter().enumerate() {
            assert!(p.is_finite(), "particle {i} went non-finite: {p:?}");
            assert!(p.length() < 100.0, "particle {i} flew away to {p:?}");
        }
    }

    use wgpu::util::DeviceExt;

    /// A compute shader must run and its output be readable back — the foundation
    /// every later GPU kernel relies on. Skips cleanly if no GPU is available.
    #[test]
    fn compute_dispatch_and_readback_work() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU available; skipping GPU compute test");
            return;
        };

        // Input data; the shader will double each value.
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let bytes = std::mem::size_of_val(&input[..]) as u64;

        let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("double"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < arrayLength(&data)) {
                        data[i] = data[i] * 2.0;
                    }
                }
                "#
                .into(),
            ),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("double pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("double"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(input.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
        queue.submit(Some(encoder.finish()));

        // Map the readback buffer and wait for it.
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i as f32) * 2.0, "element {i} not doubled");
        }
    }
}
