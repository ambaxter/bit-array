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

#![cfg(all(test, feature = "nightly"))]

#![feature(test)]

extern crate test;
extern crate rand;
extern crate bit_array;
extern crate typenum;

pub use bit_array::BitArray;

#[path = "../src/bench.rs"]
mod bench;
