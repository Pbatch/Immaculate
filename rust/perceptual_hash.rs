//! The interface shared by the near-duplicate methods.

use image::DynamicImage;

/// A perceptual hash, used to find near-duplicate images.
///
/// Each method decides both how an image is hashed and which hashes match, so it can group
/// identical hashes or hashes within some distance of each other.
pub trait PerceptualHash: Sync {
    /// What an image hashes to.
    type Hash: Send + Sync;

    /// Hashes the image, or returns `None` if it can't be hashed, e.g. because it is empty.
    fn hash(&self, image: &DynamicImage) -> Option<Self::Hash>;

    /// Groups images whose hashes match into sets of two or more.
    ///
    /// `hashes` pairs each image's index with its hash, in ascending index order. Groups are
    /// ordered by their first index, and indices within a group are ascending.
    fn group(&self, hashes: &[(usize, &Self::Hash)]) -> Vec<Vec<usize>>;
}
