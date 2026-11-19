//! Spotting image files that end before their data does, such as interrupted downloads.
//!
//! The image crate decodes a truncated JPEG without complaint, filling in what's missing
//! with gray, so this reads the file's structure instead: a file is truncated when it ends
//! before the marker its format ends with. Data after that marker, such as the video
//! appended to motion photos, doesn't count. Truncated BMP, GIF, TIFF and lossless WebP
//! files fail to decode, and are skipped instead.

/// The bytes a PNG file starts with.
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

/// Whether `bytes`, a whole image file, ends before its format's end: before a JPEG's
/// end-of-image marker, a PNG's `IEND` chunk, or the length a WebP's RIFF header gives.
/// Files in other formats, or that aren't images, are never truncated.
pub fn is_truncated(bytes: &[u8]) -> bool {
    match bytes {
        [0xFF, 0xD8, rest @ ..] => jpeg_is_truncated(rest),
        _ if bytes.starts_with(PNG_SIGNATURE) => png_is_truncated(&bytes[PNG_SIGNATURE.len()..]),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            a,
            b,
            c,
            d,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => {
            // The RIFF size counts everything after itself
            let size = u64::from(u32::from_le_bytes([*a, *b, *c, *d]));
            (bytes.len() as u64) < size + 8
        }
        _ => false,
    }
}

/// Whether `bytes`, a JPEG file after its start-of-image marker, ends before its
/// end-of-image marker.
///
/// Segments with a length are skipped whole, so markers inside them, such as an EXIF
/// thumbnail's, are never mistaken for the image's. Anything else is searched for the next
/// marker, as decoders do: entropy-coded data only holds 0xFF as 0xFF 0x00 or before a
/// restart marker, neither of which ends the image.
fn jpeg_is_truncated(bytes: &[u8]) -> bool {
    let mut rest = bytes;
    loop {
        let Some(start) = rest.iter().position(|&byte| byte == 0xFF) else {
            return true;
        };
        // Any number of 0xFF fill bytes can come before a marker
        let Some(length) = rest[start..].iter().position(|&byte| byte != 0xFF) else {
            return true;
        };
        let marker_end = start + length + 1;
        let marker = rest[marker_end - 1];
        rest = &rest[marker_end..];
        match marker {
            // End of image
            0xD9 => return false,
            // A stuffed 0xFF in entropy-coded data, a restart, TEM or a stray start of
            // image, none of which has a length
            0x00 | 0x01 | 0xD0..=0xD8 => {}
            _ => {
                let [high, low, ..] = *rest else {
                    return true;
                };
                // The length counts its own two bytes
                let length = usize::from(u16::from_be_bytes([high, low]));
                if length > rest.len() {
                    return true;
                }
                rest = &rest[length.max(2)..];
            }
        }
    }
}

/// Whether `bytes`, a PNG file after its signature, ends before its `IEND` chunk.
fn png_is_truncated(bytes: &[u8]) -> bool {
    let mut rest = bytes;
    loop {
        let Some(&[a, b, c, d, ref kind @ ..]) = rest.first_chunk::<8>() else {
            return true;
        };
        if kind == b"IEND" {
            return false;
        }
        // Length, type, data and CRC
        let size = u64::from(u32::from_be_bytes([a, b, c, d])) + 12;
        match usize::try_from(size) {
            Ok(size) if size <= rest.len() => rest = &rest[size..],
            _ => return true,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{DynamicImage, ImageFormat, RgbImage};

    use super::*;

    /// An image of `format`, noisy so its compressed data is long and varied.
    fn encoded(format: ImageFormat) -> Vec<u8> {
        let mut random = crate::test_util::XorShift(7);
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(96, 64, |_, _| {
            let bits = random.next_u64().to_le_bytes();
            image::Rgb([bits[0], bits[1], bits[2]])
        }));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, format).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn whole_files_are_not_truncated() {
        for format in [ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::WebP] {
            assert!(!is_truncated(&encoded(format)), "{format:?}");
        }
    }

    #[test]
    fn files_cut_anywhere_are_truncated() {
        for format in [ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::WebP] {
            let bytes = encoded(format);
            // Every cut that removes the end marker, once the signature is whole: shorter
            // files aren't recognized, and fail to decode instead
            let end_marker = match format {
                ImageFormat::Png => 12,
                ImageFormat::WebP => 1,
                _ => 2,
            };
            for len in 12..=bytes.len() - end_marker {
                assert!(is_truncated(&bytes[..len]), "{format:?} cut to {len}");
            }
        }
    }

    #[test]
    fn data_after_the_end_is_ignored() {
        let mut jpeg = encoded(ImageFormat::Jpeg);
        jpeg.extend_from_slice(b"\xFF\xD8 a motion photo's video \xFF");
        assert!(!is_truncated(&jpeg));
        let mut png = encoded(ImageFormat::Png);
        png.extend_from_slice(b"trailing");
        assert!(!is_truncated(&png));
    }

    #[test]
    fn jpeg_markers_inside_segments_are_skipped() {
        // An APP1 segment holding an end-of-image marker, as an EXIF thumbnail does, then
        // fill bytes, entropy-coded data with a stuffed 0xFF and a restart, and no end
        let segment = b"\xFF\xE1\x00\x06\xFF\xD9\xFF\xD9";
        let mut jpeg = b"\xFF\xD8".to_vec();
        jpeg.extend_from_slice(segment);
        jpeg.extend_from_slice(b"\xFF\xFF\xDA\x00\x02\x12\xFF\x00\x34\xFF\xD3\x56");
        assert!(is_truncated(&jpeg));
        jpeg.extend_from_slice(b"\xFF\xD9");
        assert!(!is_truncated(&jpeg));
    }

    #[test]
    fn other_formats_and_non_images_are_not_truncated() {
        let gif = encoded(ImageFormat::Gif);
        assert!(!is_truncated(&gif[..gif.len() / 2]));
        assert!(!is_truncated(b""));
        assert!(!is_truncated(b"not an image"));
    }
}
