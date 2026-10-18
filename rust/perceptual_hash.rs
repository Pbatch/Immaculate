//! The interface shared by the near-duplicate methods.

use std::fmt::Display;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::DynamicImage;
use rayon::prelude::*;

/// A perceptual hash, used to find near-duplicate images.
///
/// Each method decides both how an image is hashed and which hashes match, so it can group
/// identical hashes or hashes within some distance of each other.
pub trait PerceptualHash: Sync {
    /// What an image hashes to.
    type Hash: Send + Sync;
    /// Why hashing failed in a way that would fail for any image, e.g. because a model
    /// couldn't run. [`Infallible`](std::convert::Infallible) for methods that can't fail.
    type Error: Display + Send;

    /// Hashes the image. Returns `Ok(None)` if this image can't be hashed, e.g. because it
    /// is empty, so it is skipped.
    fn hash(&self, image: &DynamicImage) -> Result<Option<Self::Hash>, Self::Error>;

    /// Groups images whose hashes match into sets of two or more.
    ///
    /// `hashes` pairs each image's index with its hash, in ascending index order. Groups are
    /// ordered by their first index, and indices within a group are ascending.
    fn group(&self, hashes: &[(usize, &Self::Hash)]) -> Vec<Vec<usize>>;
}

/// Groups images as [`PerceptualHash::group`] does, comparing every pair with `matches`.
///
/// `matches(a, b)` is called with the earlier image's hash first. Matches chain: if A matches
/// B and B matches C, all three are one set.
pub fn group_matching<H: Sync>(
    hashes: &[(usize, &H)],
    matches: impl Fn(&H, &H) -> bool + Sync,
) -> Vec<Vec<usize>> {
    group_with(hashes, |a, sets| {
        for b in a + 1..hashes.len() {
            if matches(hashes[a].1, hashes[b].1) {
                sets.union(a, b);
            }
        }
    })
}

/// Groups images as [`PerceptualHash::group`] does. `join_later(a, sets)` is called once
/// for each position `a` in `hashes`, in parallel, and joins `a` with the later positions it
/// matches.
///
/// Matches are joined as they are found rather than collected first, so memory doesn't grow
/// with the number of matching pairs, which is quadratic in the size of a set.
pub fn group_with<H: Sync>(
    hashes: &[(usize, &H)],
    join_later: impl Fn(usize, &UnionFind) + Sync,
) -> Vec<Vec<usize>> {
    let sets = UnionFind::new(hashes.len());
    (0..hashes.len())
        .into_par_iter()
        .for_each(|a| join_later(a, &sets));

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); hashes.len()];
    for (position, &(index, _)) in hashes.iter().enumerate() {
        groups[sets.find(position)].push(index);
    }
    // Positions are visited in ascending index order, so each group is already sorted and
    // the groups are ordered by their first index once empty ones are removed.
    let mut groups: Vec<Vec<usize>> = groups.into_iter().filter(|group| group.len() > 1).collect();
    groups.sort_unstable_by_key(|group| group[0]);
    groups
}

/// Disjoint sets of positions that threads can join concurrently, without locks.
///
/// Each set's root is its lowest position, and every position's parent is at most itself, so
/// following parents always ends at a root. Joining links the higher root to the lower one,
/// and only succeeds if the higher root is still a root, so concurrent joins can't form a
/// cycle; a join that loses a race retries from the new roots.
pub struct UnionFind {
    parent: Vec<AtomicUsize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).map(AtomicUsize::new).collect(),
        }
    }

    /// The root of the set `item` is in, pointing items on the way at their grandparents
    /// (path halving) so later finds are shorter.
    fn find(&self, mut item: usize) -> usize {
        loop {
            let parent = self.parent[item].load(Ordering::Acquire);
            if parent == item {
                return item;
            }
            let grandparent = self.parent[parent].load(Ordering::Acquire);
            // Failing just means another thread moved the item closer to the root first
            let _ = self.parent[item].compare_exchange_weak(
                parent,
                grandparent,
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
            item = grandparent;
        }
    }

    /// Joins the sets `a` and `b` are in.
    pub fn union(&self, a: usize, b: usize) {
        loop {
            let (a_root, b_root) = (self.find(a), self.find(b));
            if a_root == b_root {
                return;
            }
            let (low, high) = (a_root.min(b_root), a_root.max(b_root));
            if self.parent[high]
                .compare_exchange(high, low, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}
