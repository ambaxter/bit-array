// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.

// Copyright 2016 The bit-array developers.

//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(Gankro): BitVec and BitSet are very tightly coupled. Ideally (for
// maintenance), they should be in separate files/modules, with BitSet only
// using BitVec's public API. This will be hard for performance though, because
// `BitVec` will not want to leak its internal representation while its internal
// representation as `u32`s must be assumed for best performance.

// FIXME(tbu-): `BitVec`'s methods shouldn't be `union`, `intersection`, but
// rather `or` and `and`.

// (1) Be careful, most things can overflow here because the amount of bits in
//     memory can overflow `usize`.
// (2) Make sure that the underlying vector has no excess length:
//     E. g. `nbits == 16`, `storage.len() == 2` would be excess length,
//     because the last word isn't used at all. This is important because some
//     methods rely on it (for *CORRECTNESS*).
// (3) Make sure that the unused bits in the last word are zeroed out, again
//     other methods rely on it for *CORRECTNESS*.
// (4) `BitSet` is tightly coupled with `BitVec`, so any changes you make in
// `BitVec` will need to be reflected in `BitSet`.

//! Collections implemented with bit vectors.
//!
//! # Examples
//!
//! This is a simple example of the [Sieve of Eratosthenes][sieve]
//! which calculates prime numbers up to a given limit.
//!
//! [sieve]: http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
//!
//! ```
//! extern crate typenum;
//! # extern crate bit_array;
//! use bit_array::BitArray;
//! use typenum::{Unsigned, U10000};
//!
//! # fn main() {
//!
//! // Store the primes as a BitVec
//! let primes = {
//!     // Assume all numbers are prime to begin, and then we
//!     // cross off non-primes progressively
//!     let mut bv = BitArray::<u32, U10000>::from_elem(true);
//!
//!     // Neither 0 nor 1 are prime
//!     bv.set(0, false);
//!     bv.set(1, false);
//!
//!     for i in 2.. 1 + (U10000::to_usize() as f64).sqrt() as usize {
//!         // if i is a prime
//!         if bv[i] {
//!             // Mark all multiples of i as non-prime (any multiples below i * i
//!             // will have been marked as non-prime previously)
//!             for j in i.. {
//!                 if i * j >= U10000::to_usize() {
//!                     break;
//!                 }
//!                 bv.set(i * j, false)
//!             }
//!         }
//!     }
//!     bv
//! };
//!
//! // Simple primality tests below our max bound
//! let print_primes = 20;
//! print!("The primes below {} are: ", print_primes);
//! for x in 0..print_primes {
//!     if primes.get(x).unwrap_or(false) {
//!         print!("{} ", x);
//!     }
//! }
//! println!("");
//!
//! let num_primes = primes.iter().filter(|x| *x).count();
//! println!("There are {} primes below {}", num_primes, U10000::to_usize());
//! assert_eq!(num_primes, 1_229);
//! # }
//! ```

#![cfg_attr(all(test, feature = "nightly"), feature(test))]
#[cfg(all(test, feature = "nightly"))] extern crate test;
#[cfg(all(test, feature = "nightly"))] extern crate rand;

extern crate generic_array;
extern crate typenum;
extern crate bit_vec;

use std::cmp::Ordering;
use std::cmp;
use std::fmt;
use std::hash;
use std::iter::{Chain, Enumerate, Repeat, Skip, Take};
use std::iter::FromIterator;
use std::slice;
use std::{u8, usize};
use bit_vec::BitBlock;
use generic_array::GenericArray;
use typenum::{Unsigned, Sum, Sub1, NonZero, U8, U16, U32, U64, Quot};

type MutBlocks<'a, B> = slice::IterMut<'a, B>;
type MatchWords<'a, B> = Chain<Enumerate<Blocks<'a, B>>, Skip<Take<Enumerate<Repeat<B>>>>>;

use std::ops::*;

pub trait BitsIn {
    type Output;
}

pub type BitsInOut<A> = <A as BitsIn>::Output;

macro_rules! bitsin_prim {
    ($(($prim: ty, $bits: ty)),*) => ($(
        impl BitsIn for $prim { type Output = $bits; }
    )*)
}

bitsin_prim!(
    (u8, U8),
    (u16, U16),
    (u32, U32),
    (u64, U64)
);

#[cfg(target_pointer_width = "32")]
bitsin_prim!((usize, U32));

#[cfg(target_pointer_width = "64")]
bitsin_prim!((usize, U64));

fn reverse_bits(byte: u8) -> u8 {
    let mut result = 0;
    for i in 0..u8::bits() {
        result = result | ((byte >> i) & 1) << (u8::bits() - 1 - i);
    }
    result
}

static TRUE: bool = true;
static FALSE: bool = false;

/// The bitarray type.
///
/// # Examples
///
/// ```
/// extern crate typenum;
/// # extern crate bit_array;
/// use bit_array::BitArray;
/// use typenum::U10;
///
/// # fn main() {
/// let mut bv = BitArray::<u32, U10>::from_elem(false);
///
/// // insert all primes less than 10
/// bv.set(2, true);
/// bv.set(3, true);
/// bv.set(5, true);
/// bv.set(7, true);
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // flip all values in bitvector, producing non-primes less than 10
/// bv.negate();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // reset bitvector to empty
/// bv.clear();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
/// # }
/// ```
pub struct BitArray<B: BitsIn, NBits: Unsigned + NonZero>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    storage: GenericArray<B, 
        // based on the calculation `(nbits + U32_BITS - 1) / 32::BITS` in bit-vec
        Quot<
            Sub1<Sum<NBits, BitsInOut<B>>>
            ,BitsInOut<B>
            >
        >,
}

// FIXME(Gankro): NopeNopeNopeNopeNope (wait for IndexGet to be a thing)
impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Index<usize> for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Output = bool;

    #[inline]
    fn index(&self, i: usize) -> &bool {
        if self.get(i).expect("index out of bounds") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

/// Computes the bitmask for the final word of the vector
fn mask_for_bits<B: BitBlock>(bits: usize) -> B {
    // Note especially that a perfect multiple of U32_BITS should mask all 1s.
    (!B::zero()) >> ((B::bits() - bits % B::bits()) % B::bits())
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{

    /// Creates an empty `BitArray`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U8>::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Creates a `BitArray` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U10;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U10>::from_elem(false);
    /// assert_eq!(bv.len(), 10);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// # }
    /// ```
    pub fn from_elem(bit: bool) -> Self {
        let mut bit_array = BitArray::new();
        if bit {
            bit_array.set_all();
        }
        bit_array.fix_last_block();
        bit_array
    }

    /// Transforms a byte-vector into a `BitVec`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let bv = BitArray::<u32, U8>::from_bytes(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// # }
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let total_bits = bytes.len().checked_mul(u8::bits()).expect("capacity overflow");
        let mut bit_array = BitArray::new();
        let total_array_bits = bit_array.storage.len() * B::bits();
        assert!(total_bits < total_array_bits, "bit_array with {:?} bits cannot handle byte array of {:?} bits", total_array_bits, total_bits);
        let complete_words = bytes.len() / B::bytes();
        let extra_bytes = bytes.len() % B::bytes();

        for i in 0..complete_words {
            let mut accumulator = B::zero();
            for idx in 0..B::bytes() {
                accumulator = accumulator |
                    (B::from_byte(reverse_bits(bytes[i * B::bytes() + idx])) << (idx * 8))
            }
            *bit_array.storage.get_mut(i).unwrap() = accumulator;
        }

        if extra_bytes > 0 {
            let mut last_word = B::zero();
            for (i, &byte) in bytes[complete_words * B::bytes()..].iter().enumerate() {
                last_word = last_word |
                    (B::from_byte(reverse_bits(byte)) << (i * 8));
            }
            *bit_array.storage.last_mut().unwrap() = last_word;
        }

        bit_array
    }

    /// Creates a `BitVec` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U5;
    ///
    /// # fn main() {
    /// let bv = BitArray::<u32, U5>::from_fn(|i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// # }
    /// ```
    pub fn from_fn<F>(mut f: F) -> Self
        where F: FnMut(usize) -> bool
    {
        let mut bit_array = BitArray::from_elem(false);
        for i in 0..NBits::to_usize() {
            bit_array.set(i, f(i));
        }
        bit_array
    }

    /// Applies the given operation to the blocks of self and other, and sets
    /// self to be the result. This relies on the caller not to corrupt the
    /// last word.
    #[inline]
    fn process<F>(&mut self, other: &BitArray<B, NBits>, mut op: F) -> bool
    		where F: FnMut(B, B) -> B {
        // This could theoretically be a `debug_assert!`.
        assert_eq!(self.storage.len(), other.storage.len());
        let mut changed_bits = B::zero();
        for (a, b) in self.blocks_mut().zip(other.blocks()) {
            let w = op(*a, b);
            changed_bits = changed_bits | (*a ^ w);
            *a = w;
        }
        changed_bits != B::zero()
    }
    
    /// Iterator over mutable refs to  the underlying blocks of data.
    fn blocks_mut(&mut self) -> MutBlocks<B> {
        // (2)
        self.storage.iter_mut()
    }

    /// Iterator over the underlying blocks of data
    pub fn blocks(&self) -> Blocks<B> {
        // (2)
        Blocks{iter: self.storage.iter()}
    }

    /// Exposes the raw block storage of this BitArray
    ///
    /// Only really intended for BitSet.
    pub fn storage(&self) -> &[B] {
    	&self.storage
    }

    /// Exposes the raw block storage of this BitArray
    ///
    /// Can probably cause unsafety. Only really intended for BitSet.
    pub unsafe fn storage_mut(&mut self) -> &mut[B] {
    	&mut self.storage
    }

    /// An operation might screw up the unused bits in the last block of the
    /// `BitArray`. As per (3), it's assumed to be all 0s. This method fixes it up.
    fn fix_last_block(&mut self) {
        let extra_bits = self.len() % B::bits();
        if extra_bits > 0 {
            let mask = (B::one() << extra_bits) - B::one();
            let storage_len = self.storage.len();
            let block = &mut self.storage[storage_len - 1];
            *block = *block & mask;
        }
    }

    /// Retrieves the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let bv = BitArray::<u32, U8>::from_bytes(&[0b01100000]);
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// assert_eq!(bv.get(100), None);
    ///
    /// // Can also use array indexing
    /// assert_eq!(bv[1], true);
    /// # }
    /// ```
    #[inline]
    pub fn get(&self, i: usize) -> Option<bool> {
        if i >= NBits::to_usize() {
            return None;
        }
        let w = i / B::bits();
        let b = i % B::bits();
        self.storage.get(w).map(|&block|
            (block & (B::one() << b)) != B::zero()
        )
    }

    /// Sets the value of a bit at an index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U8>::from_elem(false);
    /// bv.set(3, true);
    /// assert_eq!(bv[3], true);
    /// # }
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, x: bool) {
        assert!(i < NBits::to_usize(), "index out of bounds: {:?} >= {:?}", i, NBits::to_usize());
        let w = i / B::bits();
        let b = i % B::bits();
        let flag = B::one() << b;
        let val = if x { self.storage[w] | flag }
                  else { self.storage[w] & !flag };
        self.storage[w] = val;
    }

    /// Sets all bits to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let before = 0b01100000;
    /// let after  = 0b11111111;
    ///
    /// let mut bv = BitArray::<u32, U8>::from_bytes(&[before]);
    /// bv.set_all();
    /// assert_eq!(bv, BitArray::<u32, U8>::from_bytes(&[after]));
    /// # }
    /// ```
    #[inline]
    pub fn set_all(&mut self) {
        for w in self.storage.deref_mut() { *w = !B::zero(); }
        self.fix_last_block();
    }

    /// Flips all bits.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let before = 0b01100000;
    /// let after  = 0b10011111;
    ///
    /// let mut bv = BitArray::<u32, U8>::from_bytes(&[before]);
    /// bv.negate();
    /// assert_eq!(bv, BitArray::<u32, U8>::from_bytes(&[after]));
    /// # }
    /// ```
    #[inline]
    pub fn negate(&mut self) {
        for w in self.storage.deref_mut() { *w = !*w; }
        self.fix_last_block();
    }

    /// Calculates the union of two bitvectors. This acts like the bitwise `or`
    /// function.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitArray::<u32, U8>::from_bytes(&[a]);
    /// let b = BitArray::<u32, U8>::from_bytes(&[b]);
    ///
    /// assert!(a.union(&b));
    /// assert_eq!(a, BitArray::<u32, U8>::from_bytes(&[res]));
    /// # }
    /// ```
    #[inline]
    pub fn union(&mut self, other: &Self) -> bool {
        self.process(other, |w1, w2| (w1 | w2))
    }

    /// Calculates the intersection of two bitvectors. This acts like the
    /// bitwise `and` function.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitArray::<u32, U8>::from_bytes(&[a]);
    /// let b = BitArray::<u32, U8>::from_bytes(&[b]);
    ///
    /// assert!(a.intersect(&b));
    /// assert_eq!(a, BitArray::<u32, U8>::from_bytes(&[res]));
    /// # }
    /// ```
    #[inline]
    pub fn intersect(&mut self, other: &Self) -> bool {
        self.process(other, |w1, w2| (w1 & w2))
    }

    /// Calculates the difference between two bitvectors.
    ///
    /// Sets each element of `self` to the value of that element minus the
    /// element of `other` at the same index. Both bitvectors must be the same
    /// length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let a_b = 0b00100100; // a - b
    /// let b_a = 0b00011010; // b - a
    ///
    /// let mut bva = BitArray::<u32, U8>::from_bytes(&[a]);
    /// let bvb = BitArray::<u32, U8>::from_bytes(&[b]);
    ///
    /// assert!(bva.difference(&bvb));
    /// assert_eq!(bva, BitArray::<u32, U8>::from_bytes(&[a_b]));
    ///
    /// let bva = BitArray::<u32, U8>::from_bytes(&[a]);
    /// let mut bvb = BitArray::<u32, U8>::from_bytes(&[b]);
    ///
    /// assert!(bvb.difference(&bva));
    /// assert_eq!(bvb, BitArray::<u32, U8>::from_bytes(&[b_a]));
    /// # }
    /// ```
    #[inline]
    pub fn difference(&mut self, other: &Self) -> bool {
        self.process(other, |w1, w2| (w1 & !w2))
    }

    /// Returns `true` if all bits are 1.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U8>::from_elem(true);
    /// assert_eq!(bv.all(), true);
    ///
    /// bv.set(1, false);
    /// assert_eq!(bv.all(), false);
    /// # }
    /// ```
    pub fn all(&self) -> bool {
        let mut last_word = !B::zero();
        // Check that every block but the last is all-ones...
        self.blocks().all(|elem| {
            let tmp = last_word;
            last_word = elem;
            tmp == !B::zero()
        // and then check the last one has enough ones
        }) && (last_word == mask_for_bits(NBits::to_usize()))
    }

    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U16;
    ///
    /// # fn main() {
    /// let bv = BitArray::<u32, U16>::from_bytes(&[0b01110100, 0b10010010]);
    /// assert_eq!(bv.iter().filter(|x| *x).count(), 7);
    /// # }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<B, NBits> {
        Iter { bit_array: self, range: 0..NBits::to_usize() }
    }


    /// Returns `true` if all bits are 0.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U8>::from_elem(false);
    /// assert_eq!(bv.none(), true);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.none(), false);
    /// # }
    /// ```
    pub fn none(&self) -> bool {
        self.blocks().all(|w| w == B::zero())
    }

    /// Returns `true` if any bit is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U8>::from_elem(false);
    /// assert_eq!(bv.any(), false);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.any(), true);
    /// # }
    /// ```
    #[inline]
    pub fn any(&self) -> bool {
        !self.none()
    }

    /// Organises the bits into bytes, such that the first bit in the
    /// `BitArray` becomes the high-order bit of the first byte. If the
    /// size of the `BitArray` is not a multiple of eight then trailing bits
    /// will be filled-in with `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::{U3, U9};
    ///
    /// # fn main() {
    /// let mut bv = BitArray::<u32, U3>::from_elem(true);
    /// bv.set(1, false);
    ///
    /// assert_eq!(bv.to_bytes(), [0b10100000]);
    ///
    /// let mut bv = BitArray::<u32, U9>::from_elem(false);
    /// bv.set(2, true);
    /// bv.set(8, true);
    ///
    /// assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    /// # }
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
    	// Oh lord, we're mapping this to bytes bit-by-bit!
        fn bit<B: BitBlock + BitsIn + Default, NBits: Unsigned + NonZero>(bit_array: &BitArray<B, NBits>, byte: usize, bit: usize) -> u8 
            where NBits: Add<<B as BitsIn>::Output>,
            <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
            <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
            <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
        {
            let offset = byte * 8 + bit;
            if offset >= NBits::to_usize() {
                0
            } else {
                (bit_array.get(offset).unwrap() as u8) << (7 - bit)
            }
        }

        let len = NBits::to_usize() / 8 +
                  if NBits::to_usize() % 8 == 0 { 0 } else { 1 };
        (0..len).map(|i|
            bit(self, i, 0) |
            bit(self, i, 1) |
            bit(self, i, 2) |
            bit(self, i, 3) |
            bit(self, i, 4) |
            bit(self, i, 5) |
            bit(self, i, 6) |
            bit(self, i, 7)
        ).collect()
    }


    /// Compares a `BitArray` to a slice of `bool`s.
    /// Both the `BitArray` and slice must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the `BitArray` and slice are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bit_array;
    /// use bit_array::BitArray;
    /// use typenum::U8;
    ///
    /// # fn main() {
    /// let bv = BitArray::<u32, U8>::from_bytes(&[0b10100000]);
    ///
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false]));
    /// # }
    /// ```
    pub fn eq_vec(&self, v: &[bool]) -> bool {
        self.iter().zip(v.iter().cloned()).all(|(b1, b2)| b1 == b2)
    }

    /// Returns the total number of bits in this vector
    #[inline]
    pub fn len(&self) -> usize { NBits::to_usize() }

    /// Clears all bits in this vector.
    #[inline]
    pub fn clear(&mut self) {
        for w in self.storage.deref_mut() { *w = B::zero(); }
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Default for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn default() -> Self { BitArray { storage: GenericArray::new() } }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> FromIterator<bool> for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn from_iter<I: IntoIterator<Item=bool>>(iter: I) -> Self {
        let mut ret: Self = Default::default();
        for (i, val) in iter.into_iter().enumerate() {
            ret.set(i, val);
        }
        ret
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Clone for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn clone(&self) -> Self {
        BitArray { storage: self.storage.clone()}
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.storage.clone_from(&source.storage);
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> PartialOrd for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Ord for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.iter();
        let mut b = other.iter();
        loop {
            match (a.next(), b.next()) {
                (Some(x), Some(y)) => match x.cmp(&y) {
                    Ordering::Equal => {}
                    otherwise => return otherwise,
                },
                (None, None) => return Ordering::Equal,
                (None, _) => return Ordering::Less,
                (_, None) => return Ordering::Greater,
            }
        }
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> fmt::Debug for BitArray<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for bit in self {
            try!(write!(fmt, "{}", if bit { 1 } else { 0 }));
        }
        Ok(())
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> hash::Hash for BitArray<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for elem in self.blocks() {
            elem.hash(state);
        }
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> cmp::PartialEq for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.blocks().zip(other.blocks()).all(|(w1, w2)| w1 == w2)
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> cmp::Eq for BitArray<B, NBits>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{}


/// An iterator for `BitArray`.
#[derive(Clone)]
pub struct Iter<'a, B: 'a + BitsIn + BitBlock + Default, NBits:'a + Unsigned + NonZero> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    bit_array: &'a BitArray<B, NBits>,
    range: Range<usize>,
}

impl<'a, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Iterator for Iter<'a, B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        // NB: indexing is slow for extern crates when it has to go through &TRUE or &FALSE
        // variables.  get is more direct, and unwrap is fine since we're sure of the range.
        self.range.next().map(|i| self.bit_array.get(i).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> DoubleEndedIterator for Iter<'a, B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_array.get(i).unwrap())
    }
}

impl<'a, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> ExactSizeIterator for Iter<'a, B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{}


impl<'a, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> IntoIterator for &'a BitArray<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Item = bool;
    type IntoIter = Iter<'a, B, NBits>;

    fn into_iter(self) -> Iter<'a, B, NBits> {
        self.iter()
    }
}

pub struct IntoIter<B: BitsIn, NBits: Unsigned + NonZero>
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    bit_array: BitArray<B, NBits>,
    range: Range<usize>,
}

impl<B: BitBlock + BitsIn + Default, NBits: Unsigned + NonZero> Iterator for IntoIter<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        self.range.next().map(|i| self.bit_array.get(i).unwrap())
    }
}

impl<B: BitBlock + BitsIn + Default, NBits: Unsigned + NonZero> DoubleEndedIterator for IntoIter<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_array.get(i).unwrap())
    }
}

impl<B: BitBlock + BitsIn + Default, NBits: Unsigned + NonZero>ExactSizeIterator for IntoIter<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> IntoIterator for BitArray<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Item = bool;
    type IntoIter = IntoIter<B, NBits>;

    fn into_iter(self) -> IntoIter<B, NBits> {
        IntoIter { bit_array: self, range: 0..NBits::to_usize() }
    }
}

/// An iterator over the blocks of a `BitArray`.
#[derive(Clone)]
pub struct Blocks<'a, B: 'a> {
    iter: slice::Iter<'a, B>,
}

impl<'a, B: BitBlock> Iterator for Blocks<'a, B> {
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().cloned()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, B: BitBlock> DoubleEndedIterator for Blocks<'a, B> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().cloned()
    }
}

impl<'a, B: BitBlock> ExactSizeIterator for Blocks<'a, B> {}



#[cfg(test)]
mod tests {
    use super::{BitArray, Iter};
    use typenum::*;

    #[test]
    fn test_to_str() {
        let eightbits = BitArray::<u32, U8>::from_elem(false);
        assert_eq!(format!("{:?}", eightbits), "00000000")
    }

    #[test]
    fn test_1_element() {
        let mut act = BitArray::<u32, U1>::from_elem(false);
        assert!(act.eq_vec(&[false]));
        assert!(act.none() && !act.all());
        act = BitArray::<u32, U1>::from_elem(true);
        assert!(act.eq_vec(&[true]));
        assert!(!act.none() && act.all());
    }

    #[test]
    fn test_2_elements() {
        let mut b = BitArray::<u32, U2>::from_elem(false);
        b.set(0, true);
        b.set(1, false);
        assert_eq!(format!("{:?}", b), "10");
        assert!(!b.none() && !b.all());
    }

    #[test]
    fn test_10_elements() {
        let mut act;
        // all 0

        act = BitArray::<u32, U10>::from_elem(false);
        assert!((act.eq_vec(
                    &[false, false, false, false, false, false, false, false, false, false])));
        assert!(act.none() && !act.all());
        // all 1

        act = BitArray::<u32, U10>::from_elem(true);
        assert!((act.eq_vec(&[true, true, true, true, true, true, true, true, true, true])));
        assert!(!act.none() && act.all());
        // mixed

        act = BitArray::<u32, U10>::from_elem(false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        assert!((act.eq_vec(&[true, true, true, true, true, false, false, false, false, false])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U10>::from_elem(false);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        act.set(8, true);
        act.set(9, true);
        assert!((act.eq_vec(&[false, false, false, false, false, true, true, true, true, true])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U10>::from_elem(false);
        act.set(0, true);
        act.set(3, true);
        act.set(6, true);
        act.set(9, true);
        assert!((act.eq_vec(&[true, false, false, true, false, false, true, false, false, true])));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_31_elements() {
        let mut act;
        // all 0

        act = BitArray::<u32, U31>::from_elem(false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitArray::<u32, U31>::from_elem(true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitArray::<u32, U31>::from_elem(false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U31>::from_elem(false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U31>::from_elem(false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U31>::from_elem(false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_32_elements() {
        let mut act;
        // all 0

        act = BitArray::<u32, U32>::from_elem(false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitArray::<u32, U32>::from_elem(true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitArray::<u32, U32>::from_elem(false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U32>::from_elem(false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U32>::from_elem(false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true, true]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U32>::from_elem(false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_33_elements() {
        let mut act;
        // all 0

        act = BitArray::<u32, U33>::from_elem(false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitArray::<u32, U33>::from_elem(true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitArray::<u32, U33>::from_elem(false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U33>::from_elem(false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U33>::from_elem(false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true, true, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitArray::<u32, U33>::from_elem(false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        act.set(31, true);
        act.set(32, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true, true, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_equal_sneaky_small() {
        let mut a = BitArray::<u32, U1>::from_elem(false);
        a.set(0, true);

        let mut b = BitArray::<u32, U1>::from_elem(true);
        b.set(0, true);

        assert_eq!(a, b);
    }

    #[test]
    fn test_equal_sneaky_big() {
        let mut a = BitArray::<u32, U100>::from_elem(false);
        for i in 0..100 {
            a.set(i, true);
        }

        let mut b = BitArray::<u32, U100>::from_elem(true);
        for i in 0..100 {
            b.set(i, true);
        }

        assert_eq!(a, b);
    }

    #[test]
    fn test_from_bytes() {
        let bit_array = BitArray::<u32, U24>::from_bytes(&[0b10110110, 0b00000000, 0b11111111]);
        let str = concat!("10110110", "00000000", "11111111");
        assert_eq!(format!("{:?}", bit_array), str);
    }

    #[test]
    fn test_to_bytes() {
        let mut bv = BitArray::<u32, U3>::from_elem(true);
        bv.set(1, false);
        assert_eq!(bv.to_bytes(), [0b10100000]);

        let mut bv = BitArray::<u32, U9>::from_elem(false);
        bv.set(2, true);
        bv.set(8, true);
        assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    }

    #[test]
    fn test_from_bools() {
        let bools = vec![true, false, true, true];
        let bit_array: BitArray<u32, U4> = bools.iter().map(|n| *n).collect();
        assert_eq!(format!("{:?}", bit_array), "1011");
    }

    #[test]
    fn test_to_bools() {
        let bools = vec![false, false, true, false, false, true, true, false];
        assert_eq!(BitArray::<u32, U8>::from_bytes(&[0b00100110]).iter().collect::<Vec<bool>>(), bools);
    }


    #[test]
    fn test_bit_array_iterator() {
        let bools = vec![true, false, true, true];
        let bit_array: BitArray<u32, U4> = bools.iter().map(|n| *n).collect();

        assert_eq!(bit_array.iter().collect::<Vec<bool>>(), bools);

        let long: Vec<_> = (0..10000).map(|i| i % 2 == 0).collect();
        let bit_array: BitArray<u32, U10000> = long.iter().map(|n| *n).collect();
        assert_eq!(bit_array.iter().collect::<Vec<bool>>(), long)
    }

    #[test]
    fn test_small_difference() {
        let mut b1 = BitArray::<u32, U3>::from_elem(false);
        let mut b2 = BitArray::<u32, U3>::from_elem(false);
        b1.set(0, true);
        b1.set(1, true);
        b2.set(1, true);
        b2.set(2, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[1]);
        assert!(!b1[2]);
    }

    #[test]
    fn test_big_difference() {
        let mut b1 = BitArray::<u32, U100>::from_elem(false);
        let mut b2 = BitArray::<u32, U100>::from_elem(false);
        b1.set(0, true);
        b1.set(40, true);
        b2.set(40, true);
        b2.set(80, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[40]);
        assert!(!b1[80]);
    }

    #[test]
    fn test_small_clear() {
        let mut b = BitArray::<u32, U14>::from_elem(true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_big_clear() {
        let mut b = BitArray::<u32, U140>::from_elem(true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_bit_array_lt() {
        let mut a = BitArray::<u32, U5>::from_elem(false);
        let mut b = BitArray::<u32, U5>::from_elem(false);

        assert!(!(a < b) && !(b < a));
        b.set(2, true);
        assert!(a < b);
        a.set(3, true);
        assert!(a < b);
        a.set(2, true);
        assert!(!(a < b) && b < a);
        b.set(0, true);
        assert!(a < b);
    }

    #[test]
    fn test_ord() {
        let mut a = BitArray::<u32, U5>::from_elem(false);
        let mut b = BitArray::<u32, U5>::from_elem(false);

        assert!(a <= b && a >= b);
        a.set(1, true);
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b.set(1, true);
        b.set(2, true);
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }


    #[test]
    fn test_small_bit_array_tests() {
        let v = BitArray::<u32, U8>::from_bytes(&[0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitArray::<u32, U8>::from_bytes(&[0b00010100]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitArray::<u32, U8>::from_bytes(&[0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_big_bit_array_tests() {
        let v = BitArray::<u32, U88>::from_bytes(&[ // 88 bits
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitArray::<u32, U88>::from_bytes(&[ // 88 bits
            0, 0, 0b00010100, 0,
            0, 0, 0, 0b00110100,
            0, 0, 0]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitArray::<u32, U88>::from_bytes(&[ // 88 bits
            0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }


    #[test]
    fn test_into_iter() {
        let bools = vec![true, false, true, true];
        let bit_array: BitArray<u32, U4> = bools.iter().map(|n| *n).collect();
        let mut iter = bit_array.into_iter();
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(false), iter.next());
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(true), iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());

        let bit_array: BitArray<u32, U4> = bools.iter().map(|n| *n).collect();
        let mut iter = bit_array.into_iter();
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(false), iter.next_back());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(None, iter.next_back());
        assert_eq!(None, iter.next_back());

        let bit_array: BitArray<u32, U4> = bools.iter().map(|n| *n).collect();
        let mut iter = bit_array.into_iter();
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(false), iter.next());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next_back());
    }

    #[test]
    fn iter() {
        let b: BitArray<u32, U10> = BitArray::new();
        let _a: Iter<u32, U10> = b.iter();
    }
    
}

#[cfg(all(test, feature = "nightly"))] mod bench;
