//! Big 3-D arrows for drawing vectors (used by the educational mode).
//!
//! Each arrow is a unit shaft-and-head mesh aligned with +Y; per-instance data
//! places it at a start point, points it along a direction, and scales its length
//! and thickness. Arrows are drawn flat-coloured and always on top, so they read
//! clearly as diagram vectors over the scene.

use bytemuck::{Pod, Zeroable};

/// One arrow to draw.
///
/// What: where the arrow starts, which way it points, how long and thick it is,
/// and its colour.
/// How/why: the vertex shader orients the shared arrow mesh along `dir` and scales
/// it, so we can draw any number of labelled vectors with one mesh.
/// Units: `start` in AU; `dir` a (unit) direction; `length`/`thickness` in AU;
/// `color` linear RGBA.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ArrowInstance {
    pub start: [f32; 3],
    pub length: f32,
    pub dir: [f32; 3],
    pub thickness: f32,
    pub color: [f32; 4],
}

/// The arrow shader (WGSL): orients the mesh along the instance direction.
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,   // unit arrow along +Y
    @location(1) i_start: vec3<f32>,
    @location(2) i_length: f32,
    @location(3) i_dir: vec3<f32>,
    @location(4) i_thickness: f32,
    @location(5) i_color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    // Build an orthonormal frame with `up` along the arrow direction.
    let up = normalize(in.i_dir);
    var refv = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(up.y) > 0.99) {
        refv = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(refv, up));
    let fwd = cross(up, right);

    let m = in.position;
    let world = in.i_start
        + right * (m.x * in.i_thickness)
        + up * (m.y * in.i_length)
        + fwd * (m.z * in.i_thickness);

    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(world, 1.0);
    out.color = in.i_color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Build the unit arrow mesh (a shaft cylinder + a head cone) along +Y.
///
/// What: returns a triangle-list of positions for one arrow of length 1.
/// How/why: a `sides`-sided cylinder from y=0 to the shaft top, then a cone (with
/// a base disc) up to the tip at y=1. Radii are in "thickness units" — the shaft
/// is 1, the head is wider — and get scaled per instance.
/// Units: dimensionless mesh coordinates.
fn generate_arrow(sides: u32) -> Vec<[f32; 3]> {
    let shaft_top = 0.74f32;
    let shaft_r = 1.0f32;
    let head_r = 2.6f32;
    let tip = 1.0f32;
    let mut v = Vec::new();
    let tau = std::f32::consts::TAU;
    for i in 0..sides {
        let a0 = tau * (i as f32) / (sides as f32);
        let a1 = tau * ((i + 1) as f32) / (sides as f32);
        let (s0, c0) = a0.sin_cos();
        let (s1, c1) = a1.sin_cos();

        // Shaft side (two triangles).
        let p00 = [shaft_r * c0, 0.0, shaft_r * s0];
        let p10 = [shaft_r * c1, 0.0, shaft_r * s1];
        let p01 = [shaft_r * c0, shaft_top, shaft_r * s0];
        let p11 = [shaft_r * c1, shaft_top, shaft_r * s1];
        v.extend_from_slice(&[p00, p10, p11, p00, p11, p01]);

        // Head base disc.
        let b0 = [head_r * c0, shaft_top, head_r * s0];
        let b1 = [head_r * c1, shaft_top, head_r * s1];
        v.extend_from_slice(&[[0.0, shaft_top, 0.0], b1, b0]);

        // Head cone side.
        v.extend_from_slice(&[b0, b1, [0.0, tip, 0.0]]);
    }
    v
}

/// GPU resources for drawing arrows.
///
/// What: the arrow pipeline, the shared mesh, and a per-frame instance buffer.
/// How/why: built once; each frame we upload the current arrows and draw them all
/// in one instanced call, always on top of the scene.
/// Units: not applicable (GPU handles).
pub struct ArrowPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    instance_buf: wgpu::Buffer,
    capacity: u32,
}

impl ArrowPass {
    /// Build the arrow pipeline and mesh.
    ///
    /// What: compiles the arrow shader, makes the unit arrow, and reserves the
    /// instance buffer.
    /// How/why: flat colour, alpha blending, no culling (arrows seen from any
    /// side), and the depth test set to always-pass with no writing so arrows draw
    /// over the scene as clear diagram overlays.
    /// Units: `capacity` is a number of arrows.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let mesh = generate_arrow(16);
        let vertex_count = mesh.len() as u32;
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("arrow mesh"),
            contents: bytemuck::cast_slice(&mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arrow instances"),
            size: (capacity as usize * std::mem::size_of::<ArrowInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("arrow shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("arrow pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const MESH_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            1 => Float32x3, 2 => Float32, 3 => Float32x3, 4 => Float32, 5 => Float32x4
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("arrow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &MESH_ATTRS,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ArrowInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &INSTANCE_ATTRS,
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buf,
            vertex_count,
            instance_buf,
            capacity,
        }
    }

    /// Upload this frame's arrows.
    ///
    /// What: copies the arrow instances to the GPU.
    /// How/why: the buffer is pre-sized; we copy at most what fits.
    /// Units: see [`ArrowInstance`].
    pub fn upload(&self, queue: &wgpu::Queue, arrows: &[ArrowInstance]) {
        let count = (arrows.len() as u32).min(self.capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&arrows[..count]));
    }

    /// Record the arrow draw.
    ///
    /// What: draws the arrow mesh once per arrow instance.
    /// How/why: one instanced draw covers all the vectors.
    /// Units: `count` is a number of arrows.
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
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.draw(0..self.vertex_count, 0..count);
    }
}
