//! The 3-D renderer: turns body positions and trails into pixels.
//!
//! This module owns the GPU "scene": a shared block of camera data (the
//! [`Globals`]) plus two drawing passes — one for the shaded body spheres
//! ([`sphere::BodyPass`]) and one for the fading trails ([`trails::TrailPass`]).
//! Keeping all the GPU plumbing here leaves the maths modules (camera, ephemeris)
//! free of graphics code, as the house rules ask.

pub mod arrows;
pub mod camera;
pub mod grid;
pub mod logscale;
pub mod rings;
pub mod screenshot;
pub mod sphere;
pub mod starfield;
pub mod textures;
pub mod trails;
pub mod viewpoints;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use arrows::{ArrowInstance, ArrowPass};
use grid::{LinePass, LineSeg};
use rings::{RingPass, RingVertex};
use sphere::{BodyPass, Instance};
use starfield::{StarInstance, StarfieldPass};
use trails::{TrailPass, TrailVertex};

/// Maximum number of line *segments* the line pass can draw in one frame
/// (the adaptive 3-D grid across all levels, plus the horizon).
const LINE_CAPACITY: u32 = 16384;

/// Camera/scene data shared by every shader, as one uniform block.
///
/// What: the view-projection matrix, the Sun's position for lighting, and the
/// viewport size in pixels (used to give grid lines a real screen-space width).
/// How/why: shaders read this to place vertices on screen and to light bodies;
/// the `_pad` fields keep the layout matching WGSL's 16-byte alignment rules.
/// Units: `view_proj` dimensionless; `sun_pos` in AU (floating-origin frame);
/// `viewport` in pixels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    sun_pos: [f32; 3],
    _pad0: f32,
    viewport: [f32; 2],
    grid_fade: [f32; 2],
}

/// The complete 3-D scene renderer.
///
/// What: bundles the shared camera buffer and the body and trail passes.
/// How/why: one `Scene` is created after the GPU is ready and then driven once per
/// frame via [`Scene::render`].
/// Units: not applicable (GPU handles).
pub struct Scene {
    globals_buf: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    body_pass: BodyPass,
    trail_pass: TrailPass,
    line_pass: LinePass,
    star_pass: StarfieldPass,
    ring_pass: RingPass,
    arrow_pass: ArrowPass,
}

impl Scene {
    /// Build the scene renderer.
    ///
    /// What: creates the shared camera uniform and the two drawing passes.
    /// How/why: we make one bind-group layout for the [`Globals`] buffer and share
    /// it with both pipelines, so a single camera update feeds everything.
    /// Units: `body_count` is the maximum number of bodies (and trails); `stars`
    /// are the (static) background stars.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        body_count: u32,
        trail_len: u32,
        body_layers: &[Vec<u8>],
        stars: &[StarInstance],
    ) -> Self {
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals layout"),
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

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals bind group"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let body_pass = BodyPass::new(
            device,
            queue,
            &globals_layout,
            color_format,
            depth_format,
            body_count,
            body_layers,
        );
        let trail_pass = TrailPass::new(
            device,
            &globals_layout,
            color_format,
            depth_format,
            body_count,
            trail_len,
        );
        let line_pass = LinePass::new(
            device,
            &globals_layout,
            color_format,
            depth_format,
            LINE_CAPACITY,
        );
        let star_pass = StarfieldPass::new(device, color_format, depth_format, stars);
        let ring_pass = RingPass::new(device, queue, &globals_layout, color_format, depth_format);
        let arrow_pass = ArrowPass::new(device, &globals_layout, color_format, depth_format, 64);

        Self {
            globals_buf,
            globals_bind_group,
            body_pass,
            trail_pass,
            line_pass,
            star_pass,
            ring_pass,
            arrow_pass,
        }
    }

    /// Draw the whole scene for one frame.
    ///
    /// What: clears the screen and draws the trails and bodies.
    /// How/why: we update the camera uniform, upload this frame's instances and
    /// trail vertices, then run a single render pass that first clears colour and
    /// depth, draws the (depth-tested but non-writing) trails, and finally draws
    /// the solid bodies on top.
    /// Units: `view_proj` dimensionless; `sun_pos` in AU; `viewport` in pixels;
    /// the slices describe the bodies, trails and line segments to draw.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        view_proj: Mat4,
        sun_pos: Vec3,
        viewport: [f32; 2],
        grid_fade: [f32; 2],
        star_view_proj: Mat4,
        show_stars: bool,
        instances: &[Instance],
        trail_vertices: &[TrailVertex],
        trail_ranges: &[(u32, u32)],
        line_segs: &[LineSeg],
        ring_verts: &[RingVertex],
        arrow_instances: &[ArrowInstance],
    ) {
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            sun_pos: [sun_pos.x, sun_pos.y, sun_pos.z],
            _pad0: 0.0,
            viewport,
            grid_fade,
        };
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        self.body_pass.upload(queue, instances);
        self.trail_pass.upload(queue, trail_vertices);
        self.line_pass.upload(queue, line_segs);
        self.ring_pass.upload(queue, ring_verts);
        self.arrow_pass.upload(queue, arrow_instances);
        if show_stars {
            self.star_pass.update(queue, star_view_proj, viewport);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.02,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Stars are the backdrop, then grid/horizon, then trails, then bodies.
        if show_stars {
            self.star_pass.record(&mut pass);
        }
        self.line_pass
            .record(&mut pass, &self.globals_bind_group, line_segs.len() as u32);
        self.trail_pass
            .record(&mut pass, &self.globals_bind_group, trail_ranges);
        self.body_pass
            .record(&mut pass, &self.globals_bind_group, instances.len() as u32);
        // Saturn's rings are translucent, so they come after the solid bodies.
        self.ring_pass
            .record(&mut pass, &self.globals_bind_group, ring_verts.len() as u32);
        // Vector arrows (educational mode) draw last, always on top.
        self.arrow_pass.record(
            &mut pass,
            &self.globals_bind_group,
            arrow_instances.len() as u32,
        );
    }
}
