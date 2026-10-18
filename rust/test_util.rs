//! Helpers shared by the unit tests.

use crate::pdq::Bits;

/// xorshift64, for reproducible test data without a dependency.
pub struct XorShift(pub u64);

impl XorShift {
    pub const fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A random 256-bit hash.
    pub const fn bits(&mut self) -> Bits {
        [
            self.next_u64(),
            self.next_u64(),
            self.next_u64(),
            self.next_u64(),
        ]
    }
}
