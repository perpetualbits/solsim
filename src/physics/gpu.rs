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
//! This module currently holds only that foundation: a tiny helper to grab a
//! headless device and a smoke test that a compute shader runs and its result can
//! be read back. The tree kernels build on top of it.
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("nbody compute device"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
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
