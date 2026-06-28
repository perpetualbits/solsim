//! Drawing the bright-star background.
//!
//! Each star is a fixed direction on the sky drawn as a small round, coloured
//! glow. The stars sit on a far-away background that ignores camera *position*
//! (you cannot fly up to a star) but follows camera *rotation*, so the
//! constellations turn correctly — especially in the Earth-surface view, where
//! they rise and set as time passes.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

/// One star, ready for the GPU.
///
/// What: a sky direction, a dot size, and a colour.
/// How/why: the direction is a unit vector in the ecliptic frame; the renderer
/// projects it with rotation only, then draws a `size`-pixel glow in `color`.
/// Units: `dir` dimensionless unit vector; `size` in pixels; `color` linear RGB.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StarInstance {
    pub dir: [f32; 3],
    pub size: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// The camera data the star shader needs, as one uniform block.
///
/// What: a rotation-only view-projection matrix and the viewport size.
/// How/why: rotation-only so stars stay infinitely far (zoom and pan do not move
/// them); the viewport lets us size each star in pixels.
/// Units: `view_proj` dimensionless; `viewport` in pixels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StarUniform {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

/// The star shader (WGSL): one round glowing sprite per star.
const SHADER: &str = r#"
struct StarUniform {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: StarUniform;

// The six corners of the sprite quad (two triangles).
var<private> CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

struct VsIn {
    @location(0) dir: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec3<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, in: VsIn) -> VsOut {
    let corner = CORNERS[vi];
    var out: VsOut;
    out.corner = corner;
    out.color = in.color;

    let clip_c = u.view_proj * vec4<f32>(in.dir, 1.0);
    if (clip_c.w <= 0.0) {
        // Behind the camera: push off-screen.
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let ndc = clip_c.xy / clip_c.w;
    // Half-size in pixels → NDC offset (per axis, so the dot stays round).
    let offset = corner * in.size / u.viewport;
    // Draw at the far depth, w = 1 (screen-space sprite).
    out.clip = vec4<f32>(ndc + offset, 0.999, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.corner);
    if (r > 1.0) {
        discard;
    }
    // Soft round falloff, bright in the centre.
    let a = smoothstep(1.0, 0.2, r);
    return vec4<f32>(in.color * a, a);
}
"#;

/// GPU resources for drawing the star background.
///
/// What: the star pipeline, the (static) per-star instance buffer, and the camera
/// uniform.
/// How/why: the stars never change, so their instance buffer is filled once; only
/// the small uniform (the rotation matrix) is updated each frame.
/// Units: not applicable (GPU handles).
pub struct StarfieldPass {
    pipeline: wgpu::RenderPipeline,
    instance_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    count: u32,
}

impl StarfieldPass {
    /// Build the star pipeline and upload the stars.
    ///
    /// What: compiles the star shader, stores the star instances, and sets up the
    /// camera uniform.
    /// How/why: additive blending (colours add up) makes stars glow on the dark
    /// sky; the depth test is set to always-pass with no writing, so stars sit
    /// behind everything without disturbing the depth buffer.
    /// Units: `stars` carry pixel sizes and unit directions.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        stars: &[StarInstance],
    ) -> Self {
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("star instances"),
            contents: bytemuck::cast_slice(stars),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("star uniform"),
            size: std::mem::size_of::<StarUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("star layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("star bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("star shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("star pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("star pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<StarInstance>() as u64,
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
                    // Additive blending: overlapping stars add up and glow.
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
            uniform_buf,
            bind_group,
            count: stars.len() as u32,
        }
    }

    /// Update the rotation-only camera for this frame.
    ///
    /// What: writes the star uniform with the current view rotation and viewport.
    /// How/why: only the rotation changes per frame; the star geometry stays put.
    /// Units: `view_proj` dimensionless; `viewport` in pixels.
    pub fn update(&self, queue: &wgpu::Queue, view_proj: Mat4, viewport: [f32; 2]) {
        let u = StarUniform {
            view_proj: view_proj.to_cols_array_2d(),
            viewport,
            _pad: [0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    /// Record the draw command for all stars.
    ///
    /// What: draws six vertices (a sprite quad) per star instance.
    /// How/why: one instanced draw covers the whole sky.
    /// Units: none.
    pub fn record<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instance_buf.slice(..));
        pass.draw(0..6, 0..self.count);
    }
}
