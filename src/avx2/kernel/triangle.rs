#![allow(unused)]
use proc_macro;
use quote::{format_ident, quote};
use syn::{Expr, Result, Token, parse_macro_input};
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};
struct KernelArgs {
    xptr: Expr,
    yptr: Expr,
    tptr: Expr,
    m: Expr,
    p: Expr,
    n: Expr,
    s_x: Expr,
    s_y: Expr,
    s_t: Expr,
}
#[rustfmt::skip]
impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut args = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
        let mut args = args.into_iter();
        let xptr: Expr = args.next().ok_or(input.error("xptr not viable"))?;
        let yptr: Expr = args.next().ok_or(input.error("yptr not viable"))?;
        let tptr: Expr = args.next().ok_or(input.error("tptr not viable"))?;
        let m: Expr = args.next().ok_or(input.error("m not viable"))?;
        let p: Expr = args.next().ok_or(input.error("p not viable"))?;
        let n: Expr = args.next().ok_or(input.error("n not viable"))?;
        let s_x: Expr = args.next().ok_or(input.error("s_x not viable"))?;
        let s_y: Expr = args.next().ok_or(input.error("s_y not viable"))?;
        let s_t: Expr = args.next().ok_or(input.error("s_t not viable"))?;
        Ok(
            Self { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t, }
        )
    }
}

const M: usize = 8;
const STAG: usize = 2;
fn interleave<T> (data:&mut [T]) {
    let h = data.len() >> 1;
    for i in (1..h).step_by(2) {
        data.swap(i, i + h);
    }
}
fn riffle<T> (data: &mut [T]) {
    // this won't work b/c it will start overlapping things oddly
    let mut seen: u64 = 0;
    let m = data.len() - 1;
    for i in 1..m {
        let mut v = i;
        loop {
            let t = 2 * v % m;
            // already swapped
            if seen & ( 1<< t) != 0 { break; }
            // data in hand at i
            data.swap(i, t);
            seen |= 1 << v;
            v = t;
        }
    }
}

// fn riffle<T> (data: &mut [T]) {
//     // this won't work b/c it will start overlapping things oddly
//     let l = data.len();
//     let h = l >> 1;
//     for i in 1..l {
//         if i < h {
//             data.swap(i, 2 * i);
//         } else {
//             // data.swap(
//         }
//     }

// }


type IROWS = Vec<(usize, proc_macro2::Ident)>;
// irows
fn irows() -> IROWS {
    let mut idents = Vec::with_capacity(M);
    for idx in 0..M {
        idents.push((idx, format_ident!("row_{idx:?}")));
    }
    idents
}
// maybe this might wish to stagger instruction tho unsure
fn stagger() -> IROWS {
    assert!(M % STAG == 0, "stagger requires to be clean for macro");
    let mut idents = Vec::with_capacity(M);
    // outer iteration
    for odx in 0..M / STAG {
        // amount of unrolling
        for idx in (odx..M).step_by(M / STAG) {
            idents.push((idx, format_ident!("row_{idx:?}")));
        }
    }
    idents
}
fn load(irows: &IROWS, tptr: &Expr, s_t: &Expr) -> Vec<proc_macro2::TokenStream> {
    let mut loads = Vec::with_capacity(M);
    for (idx, ident) in irows {
        loads.push(quote! {
            let mut #ident = mask_load(#tptr.add(#idx * #s_t));
        });
    }
    loads
}
const BEES:usize = 2;
fn product(irows: &IROWS, xptr: &Expr, s_x: &Expr) -> Vec<proc_macro2::TokenStream> {
    let mut products = Vec::with_capacity(M);
    for (idx, ident) in irows {
        products.push(quote! {
            fma_accum!(#ident, #xptr.add(#idx * #s_x), b);
        });
    }
    products
}
fn save(irows: &IROWS, tptr: &Expr, s_t: &Expr) -> Vec<proc_macro2::TokenStream> {
    let mut saves = Vec::with_capacity(M);
    for (idx, ident) in irows {
        saves.push(quote! {
            _mm256_storeu_ps(#tptr.add(#idx * #s_t), #ident);
        });
    }
    saves
}
#[rustfmt::skip]
pub fn mult_unalligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let rows = irows();
    let load = load(&rows, &tptr, &s_t);
    let prod = product(&rows, &xptr, &s_x);
    let save = save(&rows, &tptr, &s_t);
    quote! {
        unsafe {
            #(#load)*
            for _ in 0..p {
                let b = _mm256_loadu_ps(yptr);
                #(#prod)*
            }
            #(#save)*
        }
    }
    .into()
}

// #[target_feature(enable = "avx,avx2,fma")]
// pub fn kernel_imult_simd_aligned(
//     mut xptr: *const f32,
//     mut yptr: *const f32,
//     tptr: *mut f32,
//     p: usize,
//     s_x: usize,
//     s_y: usize,
//     s_t: usize,
// ) {
//     // Sum[K] Union[I] { g^i = aik b^k }
//     // excels at processing panels of data ie 8 x K * K x 8;
//     unsafe {
//         let mut i_row = _mm256_loadu_ps(tptr);
//         let mut v_row = _mm256_loadu_ps(tptr.add(s_t * 4));
//         let mut ii_row = _mm256_loadu_ps(tptr.add(s_t));
//         let mut vi_row = _mm256_loadu_ps(tptr.add(s_t * 5));
//         let mut iii_row = _mm256_loadu_ps(tptr.add(s_t * 2));
//         let mut vii_row = _mm256_loadu_ps(tptr.add(s_t * 6));
//         let mut iv_row = _mm256_loadu_ps(tptr.add(s_t * 3));
//         let mut viii_row = _mm256_loadu_ps(tptr.add(s_t * 7));
//         for _ in 0..p >> 1 {
//             let b0 = _mm256_loadu_ps(yptr);
//             let b1 = _mm256_loadu_ps(yptr.add(s_y));
//             yptr = yptr.add(s_y + s_y);
//             // _mm_prefetch(tptr.add(s_t) as *const i8, _MM_HINT_T0);
//             fma_accum!(i_row, xptr, b0);
//             fma_accum!(v_row, xptr.add(4 * s_x + 1), b1);
//             fma_accum!(ii_row, xptr.add(s_x), b0);
//             fma_accum!(vi_row, xptr.add(5 * s_x + 1), b1);
//             fma_accum!(iii_row, xptr.add(2 * s_x), b0);
//             fma_accum!(vii_row, xptr.add(6 * s_x + 1), b1);
//             fma_accum!(iv_row, xptr.add(3 * s_x), b0);
//             fma_accum!(viii_row, xptr.add(7 * s_x + 1), b1);

//             fma_accum!(i_row, xptr.add(1), b1);
//             fma_accum!(v_row, xptr.add(4 * s_x), b0);
//             fma_accum!(ii_row, xptr.add(s_x + 1), b1);
//             fma_accum!(vi_row, xptr.add(5 * s_x), b0);
//             fma_accum!(iii_row, xptr.add(2 * s_x + 1), b1);
//             fma_accum!(vii_row, xptr.add(6 * s_x), b0);
//             fma_accum!(iv_row, xptr.add(3 * s_x + 1), b1);
//             fma_accum!(viii_row, xptr.add(7 * s_x), b0);
//             xptr = xptr.add(2);
//         }
//         if p & 1 == 1 {
//             let b = _mm256_loadu_ps(yptr);
//             fma_accum!(i_row, xptr, b);
//             fma_accum!(v_row, xptr.add(4 * s_x), b);
//             fma_accum!(ii_row, xptr.add(s_x), b);
//             fma_accum!(vi_row, xptr.add(5 * s_x), b);
//             fma_accum!(iii_row, xptr.add(2 * s_x), b);
//             fma_accum!(vii_row, xptr.add(6 * s_x), b);
//             fma_accum!(iv_row, xptr.add(3 * s_x), b);
//             fma_accum!(viii_row, xptr.add(7 * s_x), b);
//         }
//         _mm256_storeu_ps(tptr, i_row);
//         _mm256_storeu_ps(tptr.add(s_t * 4), v_row);
//         _mm256_storeu_ps(tptr.add(s_t), ii_row);
//         _mm256_storeu_ps(tptr.add(s_t * 5), vi_row);
//         _mm256_storeu_ps(tptr.add(s_t * 2), iii_row);
//         _mm256_storeu_ps(tptr.add(s_t * 6), vii_row);
//         _mm256_storeu_ps(tptr.add(s_t * 3), iv_row);
//         _mm256_storeu_ps(tptr.add(s_t * 7), viii_row);
//     }
// }
