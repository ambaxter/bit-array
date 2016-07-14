// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Copyright 2016 The bit-array developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::BitArray;
// was BENCH_BITS: usize = 1 << 14;
use typenum::{Unsigned, U32, U16384};
use rand::{Rng, weak_rng, XorShiftRng};

use test::{Bencher, black_box};

const U32_BITS: usize = 32;

fn rng() -> XorShiftRng {
    weak_rng()
}

#[bench]
fn bench_usize_small(b: &mut Bencher) {
    let mut r = rng();
    let mut bit_array = 0 as usize;
    b.iter(|| {
        for _ in 0..100 {
            bit_array |= 1 << ((r.next_u32() as usize) % U32_BITS);
        }
        black_box(&bit_array);
    });
}

#[bench]
fn bench_bit_set_big_fixed(b: &mut Bencher) {
    let mut r = rng();
    let mut bit_array = BitArray::<u32, U16384>::from_elem(false);
    b.iter(|| {
        for _ in 0..100 {
            bit_array.set((r.next_u32() as usize) % U16384::to_usize(), true);
        }
        black_box(&bit_array);
    });
}

#[bench]
fn bench_bit_set_big_variable(b: &mut Bencher) {
    let mut r = rng();
    let mut bit_array = BitArray::<u32, U16384>::from_elem(false);
    b.iter(|| {
        for _ in 0..100 {
            bit_array.set((r.next_u32() as usize) % U16384::to_usize(), r.gen());
        }
        black_box(&bit_array);
    });
}

#[bench]
fn bench_bit_set_small(b: &mut Bencher) {
    let mut r = rng();
    let mut bit_array = BitArray::<u32, U16384>::from_elem(false);
    b.iter(|| {
        for _ in 0..100 {
            bit_array.set((r.next_u32() as usize) % U16384::to_usize(), true);
        }
        black_box(&bit_array);
    });
}

#[bench]
fn bench_bit_array_big_union(b: &mut Bencher) {
    let mut b1 = BitArray::<u32, U16384>::from_elem(false);
    let b2 = BitArray::<u32, U16384>::from_elem(false);
    b.iter(|| {
        b1.union(&b2)
    })
}

#[bench]
fn bench_bit_array_small_iter(b: &mut Bencher) {
    let bit_array = BitArray::<u32, U32>::from_elem(false);
    b.iter(|| {
        let mut sum = 0;
        for _ in 0..10 {
            for pres in &bit_array {
                sum += pres as usize;
            }
        }
        sum
    })
}

#[bench]
fn bench_bit_array_big_iter(b: &mut Bencher) {
    let bit_array = BitArray::<u32, U16384>::from_elem(false);
    b.iter(|| {
        let mut sum = 0;
        for pres in &bit_array {
            sum += pres as usize;
        }
        sum
    })
}

#[bench]
fn bench_from_elem(b: &mut Bencher) {
    let bit = black_box(true);
    b.iter(|| {
        // create a BitArray and popcount it
        BitArray::<u32, U16384>::from_elem(bit).blocks()
                                   .fold(0, |acc, b| acc + b.count_ones())
    });
    b.bytes = U16384::to_usize() as u64 / 8;
}
