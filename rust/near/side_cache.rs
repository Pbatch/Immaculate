//! A cache of values computed from an image's side length.

use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

/// Most side lengths whose values are kept, bounding the cache's memory.
const MAX_ENTRIES: usize = 64;

/// Values computed once per image side length, since most folders have only a few image
/// sizes. Clears itself when full.
pub struct SideCache<K, T> {
    entries: HashMap<K, Rc<[T]>>,
}

impl<K, T> Default for SideCache<K, T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<K: Hash + Eq, T> SideCache<K, T> {
    /// The values for `side`, computed with `compute` if they aren't cached.
    pub fn get(&mut self, side: K, compute: impl FnOnce() -> Vec<T>) -> Rc<[T]> {
        if self.entries.len() >= MAX_ENTRIES && !self.entries.contains_key(&side) {
            self.entries.clear();
        }
        Rc::clone(self.entries.entry(side).or_insert_with(|| compute().into()))
    }
}
