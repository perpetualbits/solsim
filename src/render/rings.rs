//! Saturn's rings: a flat, translucent textured disc around the planet.
//!
//! The ring is a thin annulus (a flat washer shape) in Saturn's equatorial plane,
//! tilted from the ecliptic. Its colour and the gaps (like the Cassini division)
//! come from a radial strip texture, sampled by distance from the planet. We build
//! the ring geometry on the CPU each frame at Saturn's current position.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use super::textures;

/// Saturn's axial tilt used for the ring plane, in radians (≈ 26.7°).
const RING_TILT: f64 = 0.466;
/// Inner and outer ring radii as multiples of Saturn's drawn radius.
const RING_INNER: f64 = 1.2;
const RING_OUTER: f64 = 2.27;
/// How many segments around the ring (smoothness).
const RING_SEGMENTS: usize = 128;

/// One vertex of the ring mesh.
///
/// What: a position plus the radial texture coordinate.
/// How/why: `u` runs 0 at the inner edge to 1 at the outer edge, so the strip
/// texture is mapped across the ring's width.
/// Units: `position` in AU (floating-origin frame); `u` in 0..1.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RingVertex {
    pub position: [f32; 3],
    pub u: f32,
}

/// Build the ring's triangles around a centre (already in the render frame).
///
/// What: returns the ring as a triangle list around `center_rel`.
/// How/why: we sweep an angle around Saturn; at each step the ring plane direction
/// is `cosθ·e₁ + sinθ·e₂`, where `e₂` is tilted by Saturn's axial tilt, so the
/// ring is a tilted disc. Each segment is two triangles spanning the inner-to-
/// outer width, with `u` 0→1 across it.
/// Units: `center_rel` and `draw_radius` in AU; returns AU positions.
pub fn build(center_rel: DVec3, draw_radius: f64) -> Vec<RingVertex> {
    let inner = draw_radius * RING_INNER;
    let outer = draw_radius * RING_OUTER;
    let (st, ct) = RING_TILT.sin_cos();
    let e1 = DVec3::new(1.0, 0.0, 0.0);
    let e2 = DVec3::new(0.0, ct, st);

    let vertex = |r: f64, ang: f64, u: f32| {
        let dir = ang.cos() * e1 + ang.sin() * e2;
        let p = center_rel + r * dir;
        RingVertex {
            position: [p.x as f32, p.y as f32, p.z as f32],
            u,
        }
    };

    let mut verts = Vec::with_capacity(RING_SEGMENTS * 6);
    for k in 0..RING_SEGMENTS {
        let a0 = std::f64::consts::TAU * (k as f64) / (RING_SEGMENTS as f64);
        let a1 = std::f64::consts::TAU * ((k + 1) as f64) / (RING_SEGMENTS as f64);
        let i0 = vertex(inner, a0, 0.0);
        let o0 = vertex(outer, a0, 1.0);
        let i1 = vertex(inner, a1, 0.0);
        let o1 = vertex(outer, a1, 1.0);
        verts.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }
    verts
}

/// The ring shader (WGSL): samples the radial strip and blends it.
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var ring_tex: texture_2d<f32>;
@group(1) @binding(1) var ring_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) u: f32,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) u: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.u = in.u;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(ring_tex, ring_samp, vec2<f32>(in.u, 0.5));
    if (c.a < 0.02) {
        discard;
    }
    return c;
}
"#;

/// GPU resources for drawing Saturn's rings.
///
/// What: the ring pipeline, the ring strip texture, and a vertex buffer.
/// How/why: the texture is loaded once; the geometry is rebuilt and uploaded each
/// frame at Saturn's position.
/// Units: not applicable (GPU handles).
pub struct RingPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u32,
}

impl RingPass {
    /// Build the ring pipeline and load the ring texture.
    ///
    /// What: compiles the ring shader, uploads the ring strip texture, and reserves
    /// the vertex buffer.
    /// How/why: alpha blending so the gaps and the thin outer ring are see-through;
    /// no culling (the ring is viewed from both sides); depth tested but not
    /// written, so it is occluded by Saturn but does not fight other geometry.
    /// Units: not applicable.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let capacity = (RING_SEGMENTS as u32 + 1) * 6;
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ring vertices"),
            size: (capacity as usize * std::mem::size_of::<RingVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load the ring strip texture.
        let (tw, th, data) = textures::decode_rgba_sized(textures::RING_PNG);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ring texture"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tw * 4),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ring sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ring texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ring bind group"),
            layout: &tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ring shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ring pipeline layout"),
            bind_group_layouts: &[Some(globals_layout), Some(&tex_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ring pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RingVertex>() as u64,
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
            bind_group,
            capacity,
        }
    }

    /// Upload this frame's ring geometry.
    ///
    /// What: copies the ring vertices to the GPU.
    /// How/why: the buffer is pre-sized; we copy at most what fits.
    /// Units: see [`RingVertex`].
    pub fn upload(&self, queue: &wgpu::Queue, vertices: &[RingVertex]) {
        let count = (vertices.len() as u32).min(self.capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices[..count]));
    }

    /// Record the ring draw.
    ///
    /// What: draws `count` ring vertices as triangles.
    /// How/why: one draw covers the whole ring.
    /// Units: `count` is a number of vertices.
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
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..count, 0..1);
    }
}
