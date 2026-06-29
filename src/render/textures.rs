//! Loading the embedded body textures.
//!
//! The planet/Sun/Moon maps are baked into the program (one PNG each) and decoded
//! to RGBA at start-up. They are stacked into a single GPU texture *array*, one
//! layer per body, so all bodies can still be drawn in one instanced call — each
//! instance just carries which layer to sample.

/// Width of every body texture, in pixels (equirectangular, 2:1).
pub const TEX_W: u32 = 1024;
/// Height of every body texture, in pixels.
pub const TEX_H: u32 = 512;

/// The embedded textures, as `(key, PNG bytes)`. Layer 0 is reserved for a plain
/// white texture (for bodies without a map), so a body's array layer is its index
/// here **plus one**.
pub const TEXTURES: &[(&str, &[u8])] = &[
    ("sun", include_bytes!("../../assets/textures/sun.png")),
    (
        "mercury",
        include_bytes!("../../assets/textures/mercury.png"),
    ),
    ("venus", include_bytes!("../../assets/textures/venus.png")),
    ("earth", include_bytes!("../../assets/textures/earth.png")),
    ("mars", include_bytes!("../../assets/textures/mars.png")),
    (
        "jupiter",
        include_bytes!("../../assets/textures/jupiter.png"),
    ),
    ("saturn", include_bytes!("../../assets/textures/saturn.png")),
    ("uranus", include_bytes!("../../assets/textures/uranus.png")),
    (
        "neptune",
        include_bytes!("../../assets/textures/neptune.png"),
    ),
    ("moon", include_bytes!("../../assets/textures/moon.png")),
];

/// The Saturn ring texture (a radial strip with transparency).
pub const RING_PNG: &[u8] = include_bytes!("../../assets/textures/saturn_ring.png");

/// The texture-array layer holding the procedural cloud map.
///
/// What: the index of the baked cloud coverage texture.
/// How/why: `build_body_layers` stacks the white default (layer 0), then the
/// [`TEXTURES`] body maps, then appends the cloud layer — so it sits one past the
/// last body map. The renderer points the Earth cloud shell at this layer.
/// Units: a layer index.
pub fn cloud_layer() -> u32 {
    TEXTURES.len() as u32 + 1
}

/// The array layer for a texture key (`None` → the white layer 0).
///
/// What: maps a body's texture name to its layer in the texture array.
/// How/why: a linear search over [`TEXTURES`]; the result is the index plus one
/// because layer 0 is the white default.
/// Units: a layer index.
pub fn layer_of(name: &str) -> u32 {
    TEXTURES
        .iter()
        .position(|(k, _)| *k == name)
        .map(|i| i as u32 + 1)
        .unwrap_or(0)
}

/// Decode a PNG into a tightly packed RGBA8 buffer of size `TEX_W×TEX_H`.
///
/// What: turns PNG bytes into raw RGBA pixels for upload.
/// How/why: the `png` crate decodes the image; we expand RGB to RGBA (opaque) and,
/// as a safety net, if anything is the wrong size or fails to decode we return a
/// plain white image so the program never crashes on a bad asset.
/// Units: returns `TEX_W·TEX_H·4` bytes.
pub fn decode_rgba(bytes: &[u8]) -> Vec<u8> {
    let white = || vec![255u8; (TEX_W * TEX_H * 4) as usize];

    let decoder = png::Decoder::new(bytes);
    let Ok(mut reader) = decoder.read_info() else {
        return white();
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let Ok(info) = reader.next_frame(&mut buf) else {
        return white();
    };
    if info.width != TEX_W || info.height != TEX_H || info.bit_depth != png::BitDepth::Eight {
        return white();
    }

    let pixels = (TEX_W * TEX_H) as usize;
    match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(pixels * 4);
            for px in buf.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            rgba
        }
        png::ColorType::Grayscale => {
            let mut rgba = Vec::with_capacity(pixels * 4);
            for &g in &buf[..pixels.min(buf.len())] {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            rgba
        }
        _ => white(),
    }
}

/// Decode a PNG of any size into `(width, height, RGBA bytes)`.
///
/// What: like [`decode_rgba`] but keeps the image's own dimensions (used for the
/// ring strip, which is not 1024×512).
/// How/why: same decoding path; returns `(1, 1, white)` on failure.
/// Units: width/height in pixels; bytes are `width·height·4`.
pub fn decode_rgba_sized(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(bytes);
    let Ok(mut reader) = decoder.read_info() else {
        return (1, 1, vec![255; 4]);
    };
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let Ok(info) = reader.next_frame(&mut buf) else {
        return (1, 1, vec![255; 4]);
    };
    let pixels = (info.width * info.height) as usize;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut v = Vec::with_capacity(pixels * 4);
            for px in buf.chunks_exact(3) {
                v.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            v
        }
        _ => return (1, 1, vec![255; 4]),
    };
    (info.width, info.height, rgba)
}
