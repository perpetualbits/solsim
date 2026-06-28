//! Saving the current frame to a PNG file (the F12 key).
//!
//! We copy the just-rendered surface image into a buffer the CPU can read, then
//! write it out as a PNG. The only fiddly part is that the GPU pads each row of
//! the copied image to a 256-byte boundary, so we strip that padding when packing
//! the pixels.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Round `value` up to the next multiple of `align`.
///
/// What: the smallest multiple of `align` that is ≥ `value`.
/// How/why: GPU buffer copies require each row to start on a 256-byte boundary, so
/// we pad the row length to that.
/// Units: bytes.
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// Capture a rendered texture and write it to a PNG file.
///
/// What: reads the GPU texture back to the CPU and saves it as `path`.
/// How/why: we copy the texture into a row-padded buffer, wait for the GPU to
/// finish, then repack the pixels tightly (dropping the padding), convert the
/// channel order to RGBA, force the alpha opaque, and hand the result to the PNG
/// encoder. Errors are returned so the caller can report them instead of crashing.
/// Units: `width`/`height` in pixels.
pub fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes_per_pixel = 4u32;
    let unpadded = width * bytes_per_pixel;
    let padded = align_up(unpadded, 256);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot staging"),
        size: (padded as u64) * (height as u64),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("screenshot encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Map the buffer and block until the GPU has finished the copy.
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely())?;

    // Repack into tight RGBA rows, swapping channels for BGRA formats.
    let swap_rb = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let data = staging.slice(..).get_mapped_range();
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row = &data[(y * padded) as usize..];
        for x in 0..width {
            let px = &row[(x * 4) as usize..(x * 4 + 4) as usize];
            let (r, g, b) = if swap_rb {
                (px[2], px[1], px[0])
            } else {
                (px[0], px[1], px[2])
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    drop(data);
    staging.unmap();

    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&rgba)?;
    Ok(())
}
