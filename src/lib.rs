#![allow(unused)]
// TODO:
// then make the LX, async method
// do the 16 x 16 instruction ie 512 for the tower
// make the toml cfg to get cacheline size etc
// do a small test
// inspect the flamegraph to see if any hanging threads
// ie suspect like communication jam in l1-> l2
//
// value sanity start working on the LX async vision with the queue

// 1. Animate demo        ← most legible to employers
// 2. Blog redesign       ← makes everything else findable
// 3. Triangle kernel     ← 2hrs, unblocks LQ block
// 4. Trait refactor      ← important but least urgent

use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
use stellar::algebra::ndmethods::basic_mult;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
use stellar::kernel::matkerns::{kernel_lt_mult, kernel_mult};
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
use stellar::kernel::avx2::constants::MASK;
use std::arch::x86_64::{
    _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_broadcast_ss, _mm256_castpd_ps,
    _mm256_castps_pd, _mm256_fmadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_mask_load_ps,
    _mm256_permute2f128_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_unpackhi_pd, _mm256_unpackhi_ps, _mm256_unpacklo_pd, _mm256_unpacklo_ps,

};
const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// NOTE: could set these as cache sizes so threads reflect the amount of work
const LC: usize = 64;
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;

/// # pack transfers a copy of data from d to pack
/// * to inverse simply exchange d and b
/// - d ~ M(r, s)
///
/// * d: contains the source data of x sliced to begin at mc
/// * b: contains the target pack for the outer iteration loop
/// * re: size of the r-block
/// * se: size of the s-block
/// * s_b: stride of block
/// * s_d: stride of the matrix d
#[inline(always)]
fn pack(d: &[f32], b: &mut [f32], re: usize, se: usize, s_b: usize, s_d: usize) {
    unsafe {
        let mut doffset = 0;
        let mut boffset = 0;
        for _ in 0..re {
            b.get_unchecked_mut(boffset..boffset + se)
                .copy_from_slice(&d.get_unchecked(doffset..doffset + se));
            boffset += s_b;
            doffset += s_d;
        }
    }
}
use proc_macro;

#[proc_macro]
fn pack_simd_line(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args : Vec<_> = input.into_iter().collect();
    let bptr = &args[0];
    let dptr32 = &args[1];
    let block_usize : usize = parse_usize(&args[4]);
    let simd_width : usize = parse_usize(&args[5]);
    let offset = 0; 
    let len = block_size / simd_width;
    let tokens = Vec::with_capacity(len);
    for _ in 0..len {
        tokens.push(
            format!( "_mm256_storeu_ps({}.add({}), _mm256_loadu_ps({}.add({})", bptr, dptr, offset)
        );
        offset += simd_width;
    }
}
fn parse_usize(tt: &TokenStream) -> usize {
    if let TokenTree::Literal(lit) = tt {
        lit.to_string().parse().unwrap()
    } else {
        panic!("usize parsing failure");
    }
}

macro_rules! transfer_data_old {
    ($bptr:expr, $dptr:expr, $offset:expr) => {{
        _mm256_storeu_ps(
            $bptr.add($offset),
            _mm256_loadu_ps($dptr.add($offset))
        );
    }}
}
macro_rules! pack_x {
    ($bptr:expr, $dptr:expr, $s_d:expr) => {{
        let mut bptr = $bptr;
        let mut dptr = $dptr;
        for _ in 0..MC {
            pack_simd_line($bptr, $dptr, PC, SIMD_WIDTH);
            bptr = bptr.add(PC);
            dptr = dptr.add($s_d);
        }
    }}
}
macro_rules! pack_y {
    ($bptr:expr, $dptr:expr, $s_d:expr) => {{
        let mut bptr = $bptr;
        let mut dptr = $dptr;
        for _ in 0..PC {
            pack_simd_line($bptr, $dptr, PC, SIMD_WIDTH);
            bptr = bptr.add(NC);
            dptr = dptr.add($s_d);
        }
    }}
}
macro_rules! pack_t {
    ($bptr:expr, $dptr:expr, $s_d:expr) => {{
        let mut bptr = $bptr;
        let mut dptr = $dptr;
        for _ in 0..PC {
            pack_simd_line($bptr, $dptr, PC, SIMD_WIDTH);
            bptr = bptr.add(NC);
            dptr = dptr.add($s_d);
        }
    }}
}
fn main() {

    // let mut d_mat = generate_random_matrix(8, 64);
    // let mut d= d_mat.data.as_mut_ptr();
    // let mut b = vec![0f32; MC * PC].as_mut_ptr();
    // assert!(PC % SIMD_WIDTH == 0);
    // unsafe { 
    //     pack_x!(b, d, 64); 
    // }


    // test_gemm_equivalence();
}
