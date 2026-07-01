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
