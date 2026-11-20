//! Each property check's settings, read from the attributes of the matching Python class.

use pyo3::FromPyObject;

/// An image is dark when fewer than `fraction` of its pixels are at least `brightness`
/// bright, on a scale of 0 to 1.
#[derive(Clone, Copy, FromPyObject)]
pub struct Dark {
    /// Brightness, from 0 to 1, that a pixel needs to count as bright.
    #[pyo3(attribute)]
    pub brightness: f64,
    /// Fraction of pixels, from 0 to 1, that must be bright for an image not to be dark.
    #[pyo3(attribute)]
    pub fraction: f64,
}

/// An image is light when fewer than `fraction` of its pixels are at most `brightness`
/// bright, on a scale of 0 to 1.
#[derive(Clone, Copy, FromPyObject)]
pub struct Light {
    /// Brightness, from 0 to 1, that a pixel must not exceed to count as not light.
    #[pyo3(attribute)]
    pub brightness: f64,
    /// Fraction of pixels, from 0 to 1, that must not be light for an image not to be light.
    #[pyo3(attribute)]
    pub fraction: f64,
}

/// An image is blurry when its sharpness, the standard deviation of its thumbnail's
/// Laplacian, is below `sharpness`.
#[derive(Clone, Copy, FromPyObject)]
pub struct Blurry {
    /// Sharpness below which an image is blurry.
    #[pyo3(attribute)]
    pub sharpness: f64,
}

/// An image has low information when the entropy of its color histogram, in bits, is below
/// `entropy`.
#[derive(Clone, Copy, FromPyObject)]
pub struct LowInformation {
    /// Entropy, in bits, below which an image has low information.
    #[pyo3(attribute)]
    pub entropy: f64,
}

/// An image has an odd aspect ratio when its shorter side divided by its longer side is
/// below `ratio`.
#[derive(Clone, Copy, FromPyObject)]
pub struct OddAspectRatio {
    /// Shorter side over longer side below which an image's aspect ratio is odd.
    #[pyo3(attribute)]
    pub ratio: f64,
}

/// An image has an odd size when the square root of its area is more than `iqr_factor`
/// interquartile ranges outside the interquartile range of every image's.
#[derive(Clone, Copy, FromPyObject)]
pub struct OddSize {
    /// How many interquartile ranges outside the quartiles an image's size must be.
    #[pyo3(attribute)]
    pub iqr_factor: f64,
}

/// An image is grayscale when it decodes as grayscale, or every pixel's red, green and blue
/// differ by at most `tolerance`.
#[derive(Clone, Copy, FromPyObject)]
pub struct Grayscale {
    /// Largest difference between a pixel's channels for it to count as gray.
    #[pyo3(attribute)]
    pub tolerance: u8,
}

/// An image has low contrast when its RMS contrast, the standard deviation of its grayscale
/// values divided by 255, is below `contrast`.
#[derive(Clone, Copy, FromPyObject)]
pub struct LowContrast {
    /// RMS contrast, from 0 to 1, below which an image has low contrast.
    #[pyo3(attribute)]
    pub contrast: f64,
}
