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
