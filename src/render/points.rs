//! A world-space point-cloud pass, for drawing the galaxy mode's particles.
//!
//! Each particle is a tiny round, additive sprite at a real 3-D position (unlike
//! the star background, which lives at infinity). Tens of thousands of faint,
//! overlapping dots add up into the glowing sheet of a galaxy; colouring the two
//! galaxies differently makes the collision easy to read. Drawn additively with no
//! depth test — for a diffuse cloud, nearer and farther dots simply add.

use bytemuck::{Pod, Zeroable};

/// One particle to draw.
///
/// What: a world position, a pixel size, and an RGBA colour (alpha = brightness).
/// How/why: the vertex shader projects the position, then spreads a `size`-pixel
/// quad around it; the fragment shader rounds and fades it into a soft dot.
/// Units: `pos` in AU (floating-origin frame); `size` in pixels; `color` linear
/// RGBA.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointInstance {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
}

/// The point shader (WGSL): a camera-projected, pixel-sized round sprite.
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
    viewport: vec2<f32>,
    grid_fade: vec2<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

// The six corners of the sprite quad (two triangles).
var<private> CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct VsIn {
    @location(0) i_pos: vec3<f32>,
    @location(1) i_size: f32,
    @location(2) i_color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, in: VsIn) -> VsOut {
    var out: VsOut;
    let corner = CORNERS[vi];
    out.corner = corner;
    out.color = in.i_color;
    let clip = g.view_proj * vec4<f32>(in.i_pos, 1.0);
    if (clip.w <= 0.0) {
        // Behind the camera: push off-screen.
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let ndc = clip.xy / clip.w;
    // A fixed pixel size, converted to a normalised-device offset per axis.
    let offset = corner * in.i_size / g.viewport;
    out.clip = vec4<f32>(ndc + offset, 0.5, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.corner);
    let a = smoothstep(1.0, 0.0, r) * in.color.a;
    return vec4<f32>(in.color.rgb * a, a);
}
"#;

/// The resident point shader: positions come straight from the simulation's GPU
/// buffer (no CPU copy), and the colour/size are derived from the particle index.
///
/// The framing centre — the midpoint of the two galaxy cores — is computed in the
/// shader from `pos[0]` and `pos[n_a]`; every vertex reads the same two addresses, so
/// they stay cached and it costs essentially nothing. That is what lets the whole
/// draw skip the per-frame readback and CPU rebuild.
const RESIDENT_SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
    viewport: vec2<f32>,
    grid_fade: vec2<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct Cloud { n_a: u32, sun: u32, _b: u32, _c: u32 };
@group(1) @binding(0) var<uniform> c: Cloud;
@group(1) @binding(1) var<storage, read> pos: array<vec4<f32>>;

var<private> CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var out: VsOut;
    let corner = CORNERS[vi];
    out.corner = corner;

    // Colour/size by which galaxy (and the bright cores at 0 and n_a).
    var color = vec4<f32>(0.55, 0.72, 1.0, 0.62);   // galaxy A: cool blue
    var size = 1.8;
    if (ii == 0u || ii == c.n_a) {
        color = vec4<f32>(1.0, 1.0, 0.9, 1.0);      // core
        size = 8.0;
    } else if (ii >= c.n_a) {
        color = vec4<f32>(1.0, 0.7, 0.42, 0.62);    // galaxy B: warm
    }
    if (ii == c.sun) {
        color = vec4<f32>(1.0, 1.0, 1.0, 1.0);      // the Sun tracer: bright white
        size = 11.0;
    }
    out.color = color;

    // Keep the pair framed at the origin: subtract the midpoint of the two cores.
    let center = (pos[0].xyz + pos[c.n_a].xyz) * 0.5;
    let world = pos[ii].xyz - center;

    let clip = g.view_proj * vec4<f32>(world, 1.0);
    if (clip.w <= 0.0) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let ndc = clip.xy / clip.w;
    let offset = corner * size / g.viewport;
    out.clip = vec4<f32>(ndc + offset, 0.5, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.corner);
    let a = smoothstep(1.0, 0.0, r) * in.color.a;
    return vec4<f32>(in.color.rgb * a, a);
}
"#;

/// GPU resources for the point cloud.
///
/// What: the CPU-instanced pipeline plus a resident pipeline that draws straight from
/// the simulation's position buffer.
/// How/why: the galaxy runs on the GPU, so its points are drawn from that buffer with
/// no copy (`record_resident`); the CPU-instanced path (`upload`/`record`) remains for
/// any caller that hands over explicit points.
/// Units: not applicable (GPU handles).
pub struct PointPass {
    pipeline: wgpu::RenderPipeline,
    instance_buf: wgpu::Buffer,
    capacity: u32,
    resident_pipeline: wgpu::RenderPipeline,
    resident_layout: wgpu::BindGroupLayout,
    resident_params: wgpu::Buffer,
    resident_bind: Option<wgpu::BindGroup>,
}

impl PointPass {
    /// Build the point pipeline.
    ///
    /// What: compiles the shader and reserves the instance buffer.
    /// How/why: additive blending (dots add up into a glow), no depth test/write
    /// (a diffuse cloud need not sort), no culling.
    /// Units: `capacity` is a number of particles.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity: u32,
    ) -> Self {
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point instances"),
            size: (capacity as usize * std::mem::size_of::<PointInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x4];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &ATTRS,
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Resident pipeline: same look, but reads positions from a storage buffer and
        // colours by index. Group 0 is the shared globals; group 1 is the cloud data.
        let resident_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident points layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let resident_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident points params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let resident_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident point shader"),
            source: wgpu::ShaderSource::Wgsl(RESIDENT_SHADER.into()),
        });
        let resident_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resident point pipeline layout"),
            bind_group_layouts: &[Some(globals_layout), Some(&resident_layout)],
            immediate_size: 0,
        });
        let resident_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resident point pipeline"),
            layout: Some(&resident_pl),
            vertex: wgpu::VertexState {
                module: &resident_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resident_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            instance_buf,
            capacity,
            resident_pipeline,
            resident_layout,
            resident_params,
            resident_bind: None,
        }
    }

    /// Point the resident draw at a simulation position buffer (call when it changes).
    ///
    /// What: builds the group-1 bind group from the given `pos` storage buffer and
    /// records `n_a` (the galaxy-A/B split) for colouring.
    /// How/why: the buffer is persistent, so this is done once when galaxy mode starts;
    /// pass `None` to detach when leaving it.
    pub fn bind_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: Option<&wgpu::Buffer>,
        n_a: u32,
        sun: u32,
    ) {
        self.resident_bind = pos.map(|pos| {
            queue.write_buffer(&self.resident_params, 0, bytemuck::cast_slice(&[n_a, sun, 0, 0]));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("resident points bind"),
                layout: &self.resident_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.resident_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: pos.as_entire_binding() },
                ],
            })
        });
    }

    /// Draw `count` particles straight from the bound position buffer (six verts each).
    pub fn record_resident<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        count: u32,
    ) {
        let Some(bind) = self.resident_bind.as_ref() else {
            return;
        };
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.resident_pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_bind_group(1, bind, &[]);
        pass.draw(0..6, 0..count);
    }

    /// Upload this frame's particles (at most what fits).
    pub fn upload(&self, queue: &wgpu::Queue, points: &[PointInstance]) {
        let count = (points.len() as u32).min(self.capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&points[..count]));
    }

    /// Record the point-cloud draw (six vertices per particle).
    pub fn record<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        count: u32,
    ) {
        let count = count.min(self.capacity);
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_vertex_buffer(0, self.instance_buf.slice(..));
        pass.draw(0..6, 0..count);
    }
}
