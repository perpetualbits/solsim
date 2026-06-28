//! An adaptive 3-D grid (and a generic thick-line renderer).
//!
//! The grid is a full lattice of lines running along x, y and z — a scaffold of
//! cubes filling the space around what you are looking at. Later we will displace
//! its vertices near heavy bodies to show gravity as curvature: *mass tells space
//! how to curve, curved space tells mass how to move.*
//!
//! Three ideas keep a full 3-D grid pleasant and steady:
//! - **Distance fade.** Each line fades out with distance from where you are
//!   looking and stops entirely past a cut-off, so you see only the local lattice
//!   fading into the dark — not a solid block of lines, and no far-away clutter.
//! - **Octave levels.** We draw the grid at a spacing chosen for the zoom, plus
//!   one twice-finer level and one twice-coarser (thicker) level, cross-fading
//!   between them as you zoom so nothing pops.
//! - **Near-plane clipping.** Lines that cross behind the camera are trimmed to
//!   the near plane (in the shader) rather than dropped, so none go missing close
//!   up.
//!
//! Because real GPU lines are only one pixel wide, each line is drawn as a thin
//! screen-space rectangle ([`LinePass`]), which also lets us vary the thickness.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// Where the fade begins and ends, as multiples of the view scale.
///
/// What: the grid is full strength out to `FADE_START × view_scale` and gone by
/// `FADE_END × view_scale`.
/// How/why: tying the fade radius to the zoom keeps the sheet filling the screen
/// at every scale while never extending much past the edges.
/// Units: dimensionless multipliers of the view scale (AU).
const FADE_START: f64 = 0.3;
const FADE_END: f64 = 0.9;

/// Roughly how many base cells should span the fade diameter.
///
/// What: sets the base grid spacing for a given zoom.
/// How/why: `spacing ≈ 2·FADE_END·view_scale / IDEAL_CELLS`, then snapped to a
/// power of two so the lines sit at steady positions as you zoom. Kept modest
/// because a 3-D lattice has far more lines than a flat sheet.
/// Units: a count (dimensionless).
const IDEAL_CELLS: f64 = 5.0;

/// Largest number of cells drawn per axis, per level (in each direction).
///
/// What: caps how much lattice each level emits.
/// How/why: distance fade already hides far lines, but this stops a finer level
/// from ever exploding the line count; it is set high enough that normal zooms are
/// never actually clamped.
/// Units: a count of cells.
const MAX_CELLS: i32 = 16;

/// How many octaves away from the ideal spacing a level fades to nothing.
///
/// What: the half-width (in powers of two) of the level cross-fade window.
/// How/why: with `1.6`, the twice-finer and twice-coarser levels are partly
/// visible while levels two octaves away vanish — giving the "base + finer +
/// coarser" look the design asks for.
/// Units: octaves (powers of two).
const LEVEL_WINDOW: f64 = 1.3;

/// Line widths, in pixels, for the thinnest and thickest grid lines.
const BASE_WIDTH: f32 = 1.2;
const MIN_WIDTH: f32 = 0.4;
const MAX_WIDTH: f32 = 3.0;

/// The strongest a grid line ever gets (its alpha at full visibility).
const BASE_ALPHA: f32 = 0.35;

/// A single straight line segment to draw, in world AU coordinates.
///
/// What: two endpoints, a colour (with alpha), a pixel width, and whether it
/// should fade with distance.
/// How/why: the grid and horizon are both built as lists of these; grid lines
/// fade with distance, the horizon does not.
/// Units: `a`/`b` in AU; `color` linear RGBA in 0..1; `width` in pixels.
#[derive(Clone, Copy)]
pub struct LineSeg {
    pub a: DVec3,
    pub b: DVec3,
    pub color: [f32; 4],
    pub width: f32,
    pub fade: bool,
}

/// The distances (in AU) at which the grid starts and finishes fading.
///
/// What: returns `[start, end]` for the current zoom.
/// How/why: scales the fade constants by the view scale; the renderer hands these
/// to the shader, and the grid builder uses the end distance to bound how much it
/// generates.
/// Units: `view_scale` in AU; returns two distances in AU.
pub fn fade_distances(view_scale: f64) -> [f32; 2] {
    let v = view_scale.max(1e-12);
    [(v * FADE_START) as f32, (v * FADE_END) as f32]
}

/// Build the adaptive 3-D grid around a focus point, sized to the view.
///
/// What: returns the line segments of a cube lattice (lines along x, y and z)
/// whose spacing, brightness and thickness depend on the zoom (`view_scale`).
/// How/why: a good base spacing is about `2·FADE_END·view_scale / IDEAL_CELLS`,
/// snapped to a power of two so lines do not slide as you zoom. We draw that
/// level plus its neighbours (twice finer, twice coarser), each weighted by how
/// close its spacing is to ideal — `weight = 1 − |octaves_from_ideal| /
/// LEVEL_WINDOW` — so finer/coarser levels cross-fade smoothly. Coarser levels
/// are drawn thicker. Each level is snapped to its own lattice and only covers the
/// fade sphere (capped to [`MAX_CELLS`]); the shader then fades each line with
/// distance so the far lattice disappears, leaving a clean local scaffold.
/// Principle: this is the "infinite grid" idea at octave steps — steady under
/// zoom, full 3-D, and ready to be curved by gravity later.
/// Units: `focus` in AU; `view_scale` in AU; `base_color` linear RGB; returns
/// segments in world AU coordinates.
pub fn build_grid(focus: DVec3, view_scale: f64, base_color: [f32; 3]) -> Vec<LineSeg> {
    let mut segs = Vec::new();
    let view = view_scale.max(1e-12);
    let r_end = view * FADE_END;

    let ideal_spacing = 2.0 * r_end / IDEAL_CELLS;
    let t = ideal_spacing.log2(); // continuous "ideal level"

    let lo = (t - LEVEL_WINDOW).floor() as i32;
    let hi = (t + LEVEL_WINDOW).ceil() as i32;
    for level in lo..=hi {
        let octaves_from_ideal = level as f64 - t;
        let weight = (1.0 - octaves_from_ideal.abs() / LEVEL_WINDOW).clamp(0.0, 1.0);
        if weight <= 0.01 {
            continue;
        }
        let d = 2f64.powi(level);
        // Coarser levels (positive octaves) are drawn thicker.
        let width =
            (BASE_WIDTH * 2f32.powf(0.5 * octaves_from_ideal as f32)).clamp(MIN_WIDTH, MAX_WIDTH);
        let color = [
            base_color[0],
            base_color[1],
            base_color[2],
            BASE_ALPHA * weight as f32,
        ];

        // Cover the fade sphere, snapped to this level's lattice (so lines stay at
        // fixed world positions instead of sliding with the camera).
        let cells = ((r_end / d).ceil() as i32).clamp(1, MAX_CELLS);
        let span = cells as f64 * d;
        let cx = (focus.x / d).round() * d;
        let cy = (focus.y / d).round() * d;
        let cz = (focus.z / d).round() * d;

        let mut line = |a: DVec3, b: DVec3| {
            segs.push(LineSeg {
                a,
                b,
                color,
                width,
                fade: true,
            });
        };

        // For every lattice point in the (other two) axes, draw the full-length
        // line along each of the three directions — that makes the cubes.
        for i in -cells..=cells {
            let u = i as f64 * d;
            for j in -cells..=cells {
                let v = j as f64 * d;
                // Parallel to x, through (y = cy+u, z = cz+v).
                line(
                    DVec3::new(cx - span, cy + u, cz + v),
                    DVec3::new(cx + span, cy + u, cz + v),
                );
                // Parallel to y, through (x = cx+u, z = cz+v).
                line(
                    DVec3::new(cx + u, cy - span, cz + v),
                    DVec3::new(cx + u, cy + span, cz + v),
                );
                // Parallel to z, through (x = cx+u, y = cy+v).
                line(
                    DVec3::new(cx + u, cy + v, cz - span),
                    DVec3::new(cx + u, cy + v, cz + span),
                );
            }
        }
    }

    segs
}

/// One GPU vertex of a thick line (one corner of a screen-space rectangle).
///
/// What: both endpoints of the segment, which corner this is, and the colour.
/// How/why: the vertex shader projects both endpoints, then offsets this corner
/// sideways in pixels to give the line a real width; `params` carries
/// `(side, end, width, fade)` where `side` is ±1 (which edge), `end` is 0 or 1
/// (which endpoint), and `fade` is 1 to fade with distance or 0 not to.
/// Units: `a`/`b` in AU; `width` in pixels; `color` linear RGBA.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertexGpu {
    a: [f32; 3],
    b: [f32; 3],
    params: [f32; 4],
    color: [f32; 4],
}

/// The six (end, side) corners that make a segment's rectangle (two triangles).
const CORNERS: [(f32, f32); 6] = [
    (0.0, -1.0),
    (0.0, 1.0),
    (1.0, -1.0),
    (1.0, -1.0),
    (0.0, 1.0),
    (1.0, 1.0),
];

/// The thick-line shader (WGSL): expands each segment into a screen-space strip
/// and fades grid lines with distance from the focus.
const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    sun_pos: vec3<f32>,
    viewport: vec2<f32>,
    grid_fade: vec2<f32>, // (start distance, end distance) in AU
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct VsIn {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) params: vec4<f32>, // side, end, width, fade
    @location(3) color: vec4<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world: vec3<f32>,
    @location(2) fade_enable: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let side = in.params.x;
    let end = in.params.y;
    let width = in.params.z;

    var ca = globals.view_proj * vec4<f32>(in.a, 1.0);
    var cb = globals.view_proj * vec4<f32>(in.b, 1.0);
    var wa = in.a;
    var wb = in.b;

    var out: VsOut;
    out.color = in.color;
    out.fade_enable = in.params.w;

    let near = 0.00001;
    // Both endpoints behind the camera: nothing of this line is visible.
    if (ca.w <= near && cb.w <= near) {
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.world = wa;
        return out;
    }
    // Trim a behind-camera endpoint forward to the near plane, so a line that only
    // partly crosses behind the camera still draws its visible part (instead of
    // disappearing entirely).
    if (ca.w < near) {
        let t = (near - ca.w) / (cb.w - ca.w);
        ca = mix(ca, cb, t);
        wa = mix(wa, wb, t);
    }
    if (cb.w < near) {
        let t = (near - cb.w) / (ca.w - cb.w);
        cb = mix(cb, ca, t);
        wb = mix(wb, wa, t);
    }

    let half = globals.viewport * 0.5;
    let pa = (ca.xy / ca.w) * half; // endpoint A in pixels
    let pb = (cb.xy / cb.w) * half; // endpoint B in pixels

    var dir = pb - pa;
    let len = length(dir);
    if (len < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / len;
    }
    let nrm = vec2<f32>(-dir.y, dir.x);

    let base_clip = select(ca, cb, end > 0.5);
    let base_pix = select(pa, pb, end > 0.5);
    out.world = select(wa, wb, end > 0.5);
    let final_pix = base_pix + nrm * (width * 0.5) * side;
    let ndc = final_pix / half;

    out.clip = vec4<f32>(ndc * base_clip.w, base_clip.z, base_clip.w);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var a = in.color.a;
    if (in.fade_enable > 0.5) {
        let dist = length(in.world);
        a = a * (1.0 - smoothstep(globals.grid_fade.x, globals.grid_fade.y, dist));
    }
    if (a < 0.004) {
        discard;
    }
    return vec4<f32>(in.color.rgb, a);
}
"#;

/// GPU resources for drawing thick coloured line segments.
///
/// What: a triangle pipeline (each line is a rectangle) plus a vertex buffer.
/// How/why: built once; each frame we expand the current segments into corner
/// vertices and draw them.
/// Units: not applicable (GPU handles).
pub struct LinePass {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    capacity_segments: u32,
}

impl LinePass {
    /// Build the thick-line pipeline.
    ///
    /// What: compiles the line shader and reserves the vertex buffer.
    /// How/why: triangle-list topology (six vertices per segment), no back-face
    /// culling (the strips can face either way), alpha blending so the grid is
    /// faint, and depth testing on but no depth writing so lines sit behind the
    /// solid bodies without blocking each other.
    /// Units: `capacity_segments` is a number of line segments.
    pub fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        capacity_segments: u32,
    ) -> Self {
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line vertices"),
            size: (capacity_segments as usize * 6 * std::mem::size_of::<LineVertexGpu>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line pipeline layout"),
            bind_group_layouts: &[Some(globals_layout)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x4
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertexGpu>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRS,
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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
            capacity_segments,
        }
    }

    /// Expand the segments into corner vertices and upload them.
    ///
    /// What: turns each line segment into six vertices (a rectangle) on the GPU.
    /// How/why: the buffer is pre-sized; we copy at most what fits. Each vertex
    /// carries both endpoints so the shader can offset it sideways for width.
    /// Units: see [`LineSeg`].
    pub fn upload(&self, queue: &wgpu::Queue, segments: &[LineSeg]) {
        let count = (segments.len() as u32).min(self.capacity_segments) as usize;
        if count == 0 {
            return;
        }
        let mut verts: Vec<LineVertexGpu> = Vec::with_capacity(count * 6);
        for seg in &segments[..count] {
            let a = [seg.a.x as f32, seg.a.y as f32, seg.a.z as f32];
            let b = [seg.b.x as f32, seg.b.y as f32, seg.b.z as f32];
            let fade = if seg.fade { 1.0 } else { 0.0 };
            for (end, side) in CORNERS {
                verts.push(LineVertexGpu {
                    a,
                    b,
                    params: [side, end, seg.width, fade],
                    color: seg.color,
                });
            }
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts));
    }

    /// Record the draw command for the line segments.
    ///
    /// What: draws `segment_count × 6` vertices as triangles.
    /// How/why: one draw covers the whole grid (and horizon).
    /// Units: `segment_count` is a number of line segments.
    pub fn record<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        globals: &'p wgpu::BindGroup,
        segment_count: u32,
    ) {
        let count = segment_count.min(self.capacity_segments);
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..count * 6, 0..1);
    }
}
