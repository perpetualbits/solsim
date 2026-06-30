//! Flat translucent triangles, used to shade the swept sectors in the Kepler's
//! second-law demo (key `J`).
//!
//! Nothing clever: a per-frame list of coloured vertices drawn as a triangle list,
//! alpha-blended over the scene. The demo fills the pie-slice the Sun–planet line
//! sweeps in each equal time interval; shading them shows the slices have equal
//! area even though their shapes differ.

use bytemuck::{Pod, Zeroable};

/// One coloured triangle corner.
///
/// What: a world-space position and an RGBA colour.
/// How/why: the swept sectors are built on the CPU each frame as triangle fans from
/// the Sun, so we just hand the GPU a flat vertex list.
/// Units: `pos` in AU (floating-origin frame); `color` linear RGBA.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AreaVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

/// The area shader (WGSL): project the vertex and pass its colour through.
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// GPU resources for the translucent area triangles.
///
/// What: the pipeline and a per-frame vertex buffer.
/// How/why: built once; each frame we upload this frame's sector triangles and draw
/// them all in one call.
/// Units: not applicable (GPU handles).
pub struct AreaPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    capacity: u32,
}

impl AreaPass {
    /// Build the area pipeline.
    ///
    /// What: compiles the shader and reserves the vertex buffer.
    /// How/why: flat colour, alpha blending, no culling (the flat sectors are seen
    /// from either side) and depth test always-pass with no writing, so the shaded
    /// wedges read as a clear overlay on the orbit.
    /// Units: `capacity` is a number of vertices.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity: u32,
    ) -> Self {
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("area vertices"),
            size: (capacity as usize * std::mem::size_of::<AreaVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("area shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("area pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("area pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<AreaVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRS,
                }],
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
            capacity,
        }
    }

    /// Upload this frame's area triangles.
    ///
    /// What: copies the vertices to the GPU (at most what fits).
    /// How/why: the buffer is pre-sized; we overwrite it each frame.
    /// Units: see [`AreaVertex`].
    pub fn upload(&self, queue: &wgpu::Queue, verts: &[AreaVertex]) {
        let count = (verts.len() as u32).min(self.capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts[..count]));
    }

    /// Record the area draw.
    ///
    /// What: draws `count` vertices as a triangle list.
    /// How/why: one draw covers all the shaded sectors.
    /// Units: `count` is a number of vertices (a multiple of 3).
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
        pass.draw(0..count, 0..1);
    }
}
