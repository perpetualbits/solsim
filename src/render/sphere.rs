//! Drawing the bodies (Sun, planets, moons) as textured, shaded spheres.
//!
//! We build one unit-sphere mesh and reuse it for every body, drawing them all in
//! a single "instanced" call: the mesh is sent once, and a small per-body record
//! (its centre, size, colour tint and texture layer) is sent for each copy. All
//! the body maps live in one texture *array*, so a single draw can still paint
//! every body with its own surface.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::textures::{TEX_H, TEX_W};

/// One corner (vertex) of the sphere mesh.
///
/// What: a point on the unit sphere, its outward normal, and its texture
/// coordinate.
/// How/why: the normal drives lighting; the `uv` maps the equirectangular body map
/// onto the sphere (`u` around, `v` pole-to-pole).
/// Units: `position`/`normal` dimensionless (unit sphere); `uv` in 0..1.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

/// Per-body data sent once per drawn sphere.
///
/// What: where a body is, how big, its colour tint, whether it glows, and which
/// texture-array layer to sample.
/// How/why: instancing lets the GPU stamp the shared mesh many times, each time
/// reading one of these records. Textured bodies use a white tint (so the map
/// shows true colours); untextured bodies use the white layer and tint it.
/// Units: `center` in AU (floating-origin frame); `radius` in AU; `color` linear
/// RGB tint; `emissive` 1.0 for self-lit (the Sun) else 0.0; `tex_layer` an array
/// index.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub emissive: f32,
    pub tex_layer: u32,
    /// Axial spin: `(axis.x, axis.y, axis.z, angle)` — a (unit) rotation axis and
    /// the current spin angle in radians. Identity (no spin) is `(0,0,1,0)`.
    pub spin: [f32; 4],
}

/// The shader for textured, shaded, instanced spheres (WGSL).
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var body_tex: texture_2d_array<f32>;
@group(1) @binding(1) var body_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) i_center: vec3<f32>,
    @location(4) i_radius: f32,
    @location(5) i_color: vec3<f32>,
    @location(6) i_emissive: f32,
    @location(7) i_layer: u32,
    @location(8) i_spin: vec4<f32>,
};

// Rotate a vector around a unit axis by an angle (Rodrigues' formula).
fn rotate_axis(v: vec3<f32>, axis: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) emissive: f32,
    @location(4) uv: vec2<f32>,
    @location(5) @interpolate(flat) layer: u32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let spun = rotate_axis(in.position, in.i_spin.xyz, in.i_spin.w);
    let world = in.i_center + spun * in.i_radius;
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = rotate_axis(in.normal, in.i_spin.xyz, in.i_spin.w);
    out.color = in.i_color;
    out.emissive = in.i_emissive;
    out.uv = in.uv;
    out.layer = in.i_layer;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(body_tex, body_samp, in.uv, i32(in.layer)).rgb;
    let base = tex * in.color;
    if (in.emissive > 0.5) {
        return vec4<f32>(base, 1.0);
    }
    let n = normalize(in.normal);
    let l = normalize(globals.sun_pos - in.world);
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.06;
    let shade = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(base * shade, 1.0);
}

// Cloud shell: like `fs`, but the texture's ALPHA is the cloud coverage, so the
// fragment is translucent (alpha-blended over the planet) and lit by the Sun, so
// clouds fade into the night side at the terminator.
@fragment
fn fs_cloud(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(body_tex, body_samp, in.uv, i32(in.layer));
    let n = normalize(in.normal);
    let l = normalize(globals.sun_pos - in.world);
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.10;
    let shade = ambient + (1.0 - ambient) * diffuse;
    let col = texel.rgb * in.color * shade;
    return vec4<f32>(col, texel.a);
}
"#;

/// Generate a UV-sphere mesh (latitude/longitude grid).
///
/// What: builds the vertices and triangle indices of a unit sphere with texture
/// coordinates.
/// How/why: we walk `rings` lines of latitude and `sectors` of longitude; each
/// grid point becomes a vertex at `(cosφ·cosθ, cosφ·sinθ, sinφ)` with
/// `u = θ/2π` and `v = 1 − (latitude index)/rings` so the map's top row sits at
/// the north pole. Each cell becomes two triangles.
/// Units: returns dimensionless unit-sphere geometry; `uv` in 0..1.
fn generate_uv_sphere(sectors: u32, rings: u32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for r in 0..=rings {
        let phi = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * (r as f32) / (rings as f32);
        let (sp, cp) = phi.sin_cos();
        let v = 1.0 - (r as f32) / (rings as f32);
        for s in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sectors as f32);
            let (st, ct) = theta.sin_cos();
            let pos = [cp * ct, cp * st, sp];
            vertices.push(Vertex {
                position: pos,
                normal: pos,
                uv: [(s as f32) / (sectors as f32), v],
            });
        }
    }

    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            // Wound so the outward-facing side is the front face (counter-clockwise
            // seen from outside), so back-face culling keeps the near hemisphere we
            // actually look at — not the inside of the far one.
            indices.push(a as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((b + 1) as u16);
            indices.push(b as u16);
        }
    }

    (vertices, indices)
}

/// Upload the body texture array (one layer per body map).
///
/// What: creates a 2-D texture array and fills each layer with one decoded map.
/// How/why: stacking the maps in one array means all bodies can be drawn in a
/// single instanced call (each instance picks its layer); each layer is the same
/// `TEX_W×TEX_H` size.
/// Units: each layer is `TEX_W·TEX_H·4` bytes of RGBA.
fn upload_texture_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[Vec<u8>],
) -> wgpu::TextureView {
    let depth = layers.len().max(1) as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("body textures"),
        size: wgpu::Extent3d {
            width: TEX_W,
            height: TEX_H,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (i, data) in layers.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: i as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_W * 4),
                rows_per_image: Some(TEX_H),
            },
            wgpu::Extent3d {
                width: TEX_W,
                height: TEX_H,
                depth_or_array_layers: 1,
            },
        );
    }

    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// Holds the GPU resources needed to draw the bodies.
///
/// What: the sphere mesh buffers, the per-body instance buffer, the texture array,
/// and the pipeline.
/// How/why: created once at start-up; each frame we only refill the instance
/// buffer and issue one indexed-instanced draw.
/// Units: not applicable (GPU handles).
pub struct BodyPass {
    pipeline: wgpu::RenderPipeline,
    cloud_pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    instance_buf: wgpu::Buffer,
    cloud_instance_buf: wgpu::Buffer,
    instance_capacity: u32,
    tex_bind_group: wgpu::BindGroup,
}

impl BodyPass {
    /// Build the body-drawing pipeline, mesh and texture array.
    ///
    /// What: compiles the sphere shader, makes the mesh, uploads the body maps, and
    /// reserves space for instances.
    /// How/why: standard wgpu setup — a shader, a vertex layout (mesh data plus
    /// per-instance data), a texture-array bind group, depth testing on, and an
    /// opaque colour target.
    /// Units: `instance_capacity` is a count of bodies; `layers` are the decoded
    /// body maps.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        instance_capacity: u32,
        layers: &[Vec<u8>],
    ) -> Self {
        let (vertices, indices) = generate_uv_sphere(48, 32);
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
        // A second instance buffer for the translucent cloud shells (a handful of
        // bodies, but sized like the body buffer for headroom).
        let cloud_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud instances"),
            size: (instance_capacity as usize * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tex_view = upload_texture_array(device, queue, layers);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("body sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("body texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
        let tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("body texture bind group"),
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
            label: Some("sphere shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sphere pipeline layout"),
            bind_group_layouts: &[Some(globals_layout), Some(&tex_layout)],
            immediate_size: 0,
        });

        const MESH_ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
            3 => Float32x3, 4 => Float32, 5 => Float32x3, 6 => Float32, 7 => Uint32, 8 => Float32x4
        ];

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

        // The cloud-shell pipeline: same mesh and vertex stage, but the `fs_cloud`
        // fragment stage uses the texture's alpha as coverage, alpha-blends over the
        // planet, and does not write depth (it is a thin translucent layer drawn
        // after the solid bodies).
        let cloud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud pipeline"),
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
                // Same back-face culling as the bodies: the now-correct winding
                // keeps the near (camera-facing) hemisphere of the shell, which
                // draws over the planet, and culls the far one (no rim double-blend).
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                // LessEqual, not Less: the shell sits just in front of the planet,
                // but the huge far/near ratio (the far plane is pinned at ≥100 AU)
                // leaves so little depth precision at the planet that the shell and
                // the surface can round to the *same* depth; strict Less would then
                // reject those fragments. Nearer bodies still occlude the clouds.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cloud"),
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
            cloud_pipeline,
            vertex_buf,
            index_buf,
            index_count,
            instance_buf,
            cloud_instance_buf,
            instance_capacity,
            tex_bind_group,
        }
    }

    /// Copy this frame's body records into the instance buffer.
    ///
    /// What: uploads the per-body positions/colours/layers to the GPU.
    /// How/why: the buffer was pre-sized at start-up, so we just overwrite it.
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
    /// What: tells the GPU to draw `count` copies of the sphere mesh, textured.
    /// How/why: one indexed-instanced draw renders every body at once.
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
        pass.set_bind_group(1, &self.tex_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..count.min(self.instance_capacity));
    }

    /// Copy this frame's translucent cloud-shell records into their buffer.
    ///
    /// What: uploads the per-shell positions/sizes/layers to the GPU.
    /// How/why: a separate instance buffer from the solid bodies, since the cloud
    /// shells are drawn with a different (alpha-blended) pipeline.
    /// Units: see [`Instance`].
    pub fn upload_clouds(&self, queue: &wgpu::Queue, instances: &[Instance]) {
        let count = (instances.len() as u32).min(self.instance_capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(
            &self.cloud_instance_buf,
            0,
            bytemuck::cast_slice(&instances[..count]),
        );
    }

    /// Record the draw command for the translucent cloud shells.
    ///
    /// What: draws `count` slightly-enlarged spheres, alpha-blended over the bodies.
    /// How/why: same mesh as the bodies, but the cloud pipeline reads the texture's
    /// alpha as coverage and does not write depth, so the clouds layer correctly
    /// over the already-drawn planet. Call this after the solid bodies.
    /// Units: `count` is a number of cloud shells.
    pub fn record_clouds<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.cloud_pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_bind_group(1, &self.tex_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.cloud_instance_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..count.min(self.instance_capacity));
    }
}
