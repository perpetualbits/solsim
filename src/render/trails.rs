//! Fading trails that show where each body has been.
//!
//! For every body we keep a rolling history of its **true** positions (in the
//! Sun-centred AU frame). Each frame we draw that history as a line that fades
//! from solid at the body to transparent at the oldest point, like a comet tail.
//! Storing *true* positions means the trails stay physically honest no matter how
//! we move or (later) log-scale the view.

use std::collections::VecDeque;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// One point of a trail, ready for the GPU.
///
/// What: a position plus a colour whose alpha encodes the point's age.
/// How/why: the vertex shader just transforms the position; the alpha (faded for
/// old points) makes the tail die away smoothly.
/// Units: `position` in AU (already shifted into the floating-origin frame);
/// `color` is linear RGBA in 0..1.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TrailVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// The rolling position history of a single body.
///
/// What: a length-limited queue of recent true positions plus the trail's colour.
/// How/why: a `VecDeque` lets us cheaply push the newest point and drop the oldest
/// once we exceed `capacity` — a ring buffer in disguise.
/// Units: positions in AU; `color` is linear RGB; `capacity` a count of points.
pub struct Trail {
    points: VecDeque<DVec3>,
    color: [f32; 3],
    capacity: usize,
}

impl Trail {
    /// Create an empty trail of a given colour and length.
    ///
    /// What: makes a trail with no history yet.
    /// How/why: reserves capacity up front so pushing points never reallocates.
    /// Units: `color` is linear RGB in 0..1; `capacity` a count of points.
    pub fn new(color: [f32; 3], capacity: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(capacity + 1),
            color,
            capacity,
        }
    }

    /// Record a new position, dropping the oldest if full.
    ///
    /// What: appends the body's current true position to the history.
    /// How/why: we skip points that are essentially identical to the last one (so
    /// a paused simulation does not pile up duplicates), then trim to capacity.
    /// Units: `pos` in AU.
    pub fn record(&mut self, pos: DVec3) {
        if let Some(last) = self.points.back() {
            if last.distance_squared(pos) < 1e-18 {
                return;
            }
        }
        self.points.push_back(pos);
        while self.points.len() > self.capacity {
            self.points.pop_front();
        }
    }

    /// Remove all recorded points.
    ///
    /// What: empties the trail.
    /// How/why: used by the "clear trails" (R) key; just drops the history.
    /// Units: none.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Build the drawable vertices for this trail, relative to the camera target.
    ///
    /// What: turns the stored true positions into faded line vertices.
    /// How/why: each point is shifted by the floating-origin `target` and cast to
    /// f32; its alpha grows from 0 at the oldest point to 1 at the newest, so the
    /// tail fades. Appends to `out` and returns how many vertices were added.
    /// Units: `target` in AU; output positions in AU (floating-origin frame).
    fn append_vertices(
        &self,
        transform: &dyn Fn(DVec3) -> DVec3,
        origin_disp: DVec3,
        out: &mut Vec<TrailVertex>,
    ) -> u32 {
        let n = self.points.len();
        for (i, p) in self.points.iter().enumerate() {
            // 0.0 for the oldest point, 1.0 for the newest.
            let age = if n > 1 {
                i as f32 / (n - 1) as f32
            } else {
                1.0
            };
            let rel = (transform(*p) - origin_disp).as_vec3();
            out.push(TrailVertex {
                position: [rel.x, rel.y, rel.z],
                color: [self.color[0], self.color[1], self.color[2], age],
            });
        }
        n as u32
    }
}

/// All the bodies' trails together.
///
/// What: one [`Trail`] per body, kept in the same order as the body list.
/// How/why: grouping them lets us record and draw every trail in one place.
/// Units: positions in AU.
pub struct TrailSet {
    trails: Vec<Trail>,
}

impl TrailSet {
    /// Create a set of empty trails with the given colours and length.
    ///
    /// What: one trail per colour supplied, each holding up to `capacity` points.
    /// How/why: the caller passes the body colours so trails match their body, and
    /// the configured trail length.
    /// Units: colours are linear RGB; `capacity` a count of points.
    pub fn new(colors: &[[f32; 3]], capacity: usize) -> Self {
        Self {
            trails: colors.iter().map(|c| Trail::new(*c, capacity)).collect(),
        }
    }

    /// Record one position for each body this frame.
    ///
    /// What: appends the current true position of every body to its trail.
    /// How/why: positions must line up with the trail order; extra positions are
    /// ignored if counts ever mismatch.
    /// Units: positions in AU.
    pub fn record(&mut self, positions: &[DVec3]) {
        for (trail, pos) in self.trails.iter_mut().zip(positions.iter()) {
            trail.record(*pos);
        }
    }

    /// Erase every trail.
    ///
    /// What: clears all bodies' histories at once.
    /// How/why: convenience for the "clear trails" control.
    /// Units: none.
    pub fn clear(&mut self) {
        for trail in &mut self.trails {
            trail.clear();
        }
    }

    /// Build the trail vertices and per-trail ranges for the visible bodies.
    ///
    /// What: produces one flat vertex list plus a `(start, count)` for each visible
    /// trail.
    /// How/why: the GPU draws each trail as its own line strip, so we concatenate
    /// the vertices but remember where each run begins and ends. Bodies whose
    /// `visible` flag is false are skipped, so hidden bodies leave no trail on
    /// screen (their history is still recorded, ready for when they reappear).
    /// Units: `origin` in AU; `transform` is the display warp (identity in linear
    /// mode, logarithmic in log mode); `visible` matches the trail/body order;
    /// ranges are vertex indices/counts.
    pub fn build(
        &self,
        transform: impl Fn(DVec3) -> DVec3,
        origin: DVec3,
        visible: &[bool],
    ) -> (Vec<TrailVertex>, Vec<(u32, u32)>) {
        let transform: &dyn Fn(DVec3) -> DVec3 = &transform;
        let origin_disp = transform(origin);
        let mut verts = Vec::new();
        let mut ranges = Vec::new();
        let mut start = 0u32;
        for (i, trail) in self.trails.iter().enumerate() {
            if !visible.get(i).copied().unwrap_or(true) {
                continue;
            }
            let count = trail.append_vertices(transform, origin_disp, &mut verts);
            ranges.push((start, count));
            start += count;
        }
        (verts, ranges)
    }
}

/// The shader for the fading trail lines (WGSL).
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// GPU resources for drawing trails.
///
/// What: the line-strip pipeline and a vertex buffer big enough for all trails.
/// How/why: built once; each frame we upload the current vertices and draw each
/// trail with its own line-strip draw call so separate bodies are not joined.
/// Units: not applicable (GPU handles).
pub struct TrailPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    capacity: u32,
}

impl TrailPass {
    /// Build the trail pipeline.
    ///
    /// What: compiles the trail shader and reserves the vertex buffer.
    /// How/why: line-strip topology, alpha blending so faded points are see-
    /// through, and depth testing on (but no depth writing) so trails sit behind
    /// the solid bodies without blocking each other.
    /// Units: `body_count` × `trail_len` sets the buffer size in vertices.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        body_count: u32,
        trail_len: u32,
    ) -> Self {
        let capacity = body_count * trail_len;
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail vertices"),
            size: (capacity as usize * std::mem::size_of::<TrailVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trail shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trail pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trail pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TrailVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRS,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
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
            capacity,
        }
    }

    /// Upload this frame's trail vertices.
    ///
    /// What: copies the concatenated trail vertices to the GPU.
    /// How/why: the buffer is pre-sized; we copy at most what fits.
    /// Units: see [`TrailVertex`].
    pub fn upload(&self, queue: &wgpu::Queue, vertices: &[TrailVertex]) {
        let count = (vertices.len() as u32).min(self.capacity) as usize;
        if count == 0 {
            return;
        }
        queue.write_buffer(
            &self.vertex_buf,
            0,
            bytemuck::cast_slice(&vertices[..count]),
        );
    }

    /// Record one line-strip draw per trail.
    ///
    /// What: draws each body's trail as a separate fading polyline.
    /// How/why: a strip needs at least two points; we skip empty/one-point trails
    /// and draw each `(start, count)` range on its own so bodies are not linked.
    /// Units: ranges are vertex indices/counts.
    pub fn record<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        ranges: &[(u32, u32)],
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        for &(start, count) in ranges {
            if count >= 2 && start + count <= self.capacity {
                pass.draw(start..start + count, 0..1);
            }
        }
    }
}
