//! Drawing the bodies (Sun, Earth, Moon) as shaded spheres.
//!
//! We build one unit-sphere mesh and reuse it for every body, drawing them all in
//! a single "instanced" call: the mesh is sent once, and a small per-body record
//! (its centre, size and colour) is sent for each copy. This is fast and keeps the
//! code short.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// One corner (vertex) of the sphere mesh.
///
/// What: a point on the unit sphere together with the direction it faces.
/// How/why: on a unit sphere the outward normal equals the position, but we store
/// both so the lighting maths in the shader reads clearly.
/// Units: `position` and `normal` are dimensionless (a unit sphere); the body's
/// real size is applied later via the per-instance radius.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Per-body data sent once per drawn sphere.
///
/// What: where a body is, how big to draw it, its colour, and whether it glows.
/// How/why: instancing lets the GPU stamp the shared mesh many times, each time
/// reading one of these records to place and colour the copy.
/// Units: `center` in AU (already shifted into the camera's floating-origin
/// frame); `radius` in AU; `color` is linear RGB in 0..1; `emissive` is 1.0 for
/// self-lit bodies (the Sun) or 0.0 for lit ones (Earth, Moon).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub emissive: f32,
}

/// The shader for shaded, instanced spheres (written in WGSL).
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) i_center: vec3<f32>,
    @location(3) i_radius: f32,
    @location(4) i_color: vec3<f32>,
    @location(5) i_emissive: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let world = in.i_center + in.position * in.i_radius;
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = in.normal;
    out.color = in.i_color;
    out.emissive = in.i_emissive;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if (in.emissive > 0.5) {
        return vec4<f32>(in.color, 1.0);
    }
    let n = normalize(in.normal);
    let l = normalize(globals.sun_pos - in.world);
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.06;
    let shade = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(in.color * shade, 1.0);
}
"#;

/// Generate a UV-sphere mesh (latitude/longitude grid).
///
/// What: builds the vertices and triangle indices of a unit sphere.
/// How/why: we walk `rings` lines of latitude and `sectors` of longitude; each
/// grid point becomes a vertex at `(cosφ·cosθ, cosφ·sinθ, sinφ)` and each grid
/// cell becomes two triangles. More rings/sectors means a smoother ball.
/// Units: returns dimensionless unit-sphere geometry.
fn generate_uv_sphere(sectors: u32, rings: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for r in 0..=rings {
        // Latitude from the south pole (-π/2) to the north pole (+π/2).
        let phi = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * (r as f32) / (rings as f32);
        let (sp, cp) = phi.sin_cos();
        for s in 0..=sectors {
            // Longitude all the way around (0 to 2π).
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sectors as f32);
            let (st, ct) = theta.sin_cos();
            let pos = [cp * ct, cp * st, sp];
            vertices.push(Vertex {
                position: pos,
                normal: pos,
            });
        }
    }

    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            // Two triangles per grid cell.
            indices.push(a as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((b + 1) as u16);
        }
    }

    (vertices, indices)
}

/// Holds the GPU resources needed to draw the bodies.
///
/// What: the sphere mesh buffers, the per-body instance buffer, and the pipeline.
/// How/why: created once at start-up; each frame we only refill the instance
/// buffer with the current body positions and issue one indexed-instanced draw.
/// Units: not applicable (GPU handles).
pub struct BodyPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    instance_buf: wgpu::Buffer,
    instance_capacity: u32,
}

impl BodyPass {
    /// Build the body-drawing pipeline and mesh.
    ///
    /// What: compiles the sphere shader, makes the mesh, and reserves space for
    /// instances.
    /// How/why: standard wgpu setup — a shader, a vertex layout (mesh data plus
    /// per-instance data), depth testing on, and an opaque colour target.
    /// Units: `instance_capacity` is a count of bodies.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        instance_capacity: u32,
    ) -> Self {
        let (vertices, indices) = generate_uv_sphere(28, 18);
        let index_count = indices.len() as u32;

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body instances"),
            size: (instance_capacity as usize * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sphere shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sphere pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const MESH_ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 4] =
            wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sphere pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &MESH_ATTRS,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &INSTANCE_ATTRS,
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buf,
            index_buf,
            index_count,
            instance_buf,
            instance_capacity,
        }
    }

    /// Copy this frame's body records into the instance buffer.
    ///
    /// What: uploads the per-body positions/colours to the GPU.
    /// How/why: the buffer was pre-sized at start-up, so we just overwrite its
    /// contents; we never send more than it can hold.
    /// Units: see [`Instance`].
    pub fn upload(&self, queue: &wgpu::Queue, instances: &[Instance]) {
        let count = (instances.len() as u32).min(self.instance_capacity) as usize;
        queue.write_buffer(
            &self.instance_buf,
            0,
            bytemuck::cast_slice(&instances[..count]),
        );
    }

    /// Record the draw command for all bodies.
    ///
    /// What: tells the GPU to draw `count` copies of the sphere mesh.
    /// How/why: one indexed-instanced draw renders every body at once, each using
    /// its own instance record.
    /// Units: `count` is a number of bodies.
    pub fn record<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..count.min(self.instance_capacity));
    }
}
