#![allow(unused)]
use crate::instructs::perms::{interleave, interleave_partitions, riffle, riffle_partitions};
use proc_macro;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Result, Token, parse_macro_input};

type Vars = Vec<proc_macro2::Ident>;
type Instr = Vec<proc_macro2::TokenStream>;

const M: usize = 8;
const STAG: usize = 2;
const BEES: usize = 2;

macro_rules! parse_next {
    ($args:expr, $input:expr) => {
        $args.next().ok_or($input.error("variable not found"))?
    };
}

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
// #[rustfmt::skip]
impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut args = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
        let mut args = args.into_iter();
        Ok(Self {
            xptr: parse_next!(args, input),
            yptr: parse_next!(args, input),
            tptr: parse_next!(args, input),
            m: parse_next!(args, input),
            p: parse_next!(args, input),
            n: parse_next!(args, input),
            s_x: parse_next!(args, input),
            s_y: parse_next!(args, input),
            s_t: parse_next!(args, input),
        })
    }
}
fn irows() -> Vars {
    // let mut idents = Vec::with_capacity(M);
    // for idx in 0..M {
    //     idents.push(format_ident!("row_{idx:?}"));
    // }
    // idents
    (0..M).map( |idx | format_ident!("row_{idx:?}")).collect()
}
fn bees() -> Vars {
    let mut bees = Vec::with_capacity(BEES);
    for idx in 0..BEES {
        let bee = format_ident!("b{idx:?}");
        bees.push(bee);
    }
    bees
}
fn load_target(irows: &Vars, tptr: &Expr, s_t: &Expr) -> Instr {
    let mut loads = Vec::with_capacity(M);
    for (idx, ident) in irows.iter().enumerate() {
        loads.push(quote! {
            let mut #ident = mask_load(#tptr.add(#idx * #s_t));
        });
    }
    loads
}
fn load_yvecs(bees: &Vars, yptr: &Expr, s_y: &Expr) -> Instr {
    let mut loads = Vec::with_capacity(M);
    for (bdx, bee) in bees.iter().enumerate() {
        loads.push(quote! {
            let #bee = _mm256_loadu_ps(#yptr + #bdx * #s_y);

        });
    }
    loads
}
fn write_outcome(irows: &Vars, tptr: &Expr, s_t: &Expr) -> Instr {
    let mut saves = Vec::with_capacity(M);
    for (idx, ident) in irows.iter().enumerate() {
        saves.push(quote! {
            _mm256_storeu_ps(#tptr.add(#idx * #s_t), #ident);
        });
    }
    saves
}
fn fma_product(irows: &Vars, bees: &Vars, xptr: &Expr, s_x: &Expr) -> Instr {
    let mut products = Vec::with_capacity(BEES * M);
    for (bdx, b) in bees.iter().enumerate() {
        for (idx, ident) in irows.iter().enumerate() {
            products.push(quote! {
                fma_accum!(#ident, #xptr.add(#idx * #s_x + #bdx), #b);
            });
        }
    }
    products
}
#[rustfmt::skip]
pub fn mult_unalligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let rows = irows();
    let bees = bees();
    let mut yvecs = load_yvecs(&bees, &yptr, &s_y);
    let mut load = load_target(&rows, &tptr, &s_t);
    let mut prod = fma_product(&rows, &bees, &xptr, &s_x);
    let mut save = write_outcome(&rows, &tptr, &s_t);
    
    riffle(&mut load);
    // riffle(&mut prod);
    riffle_partitions(&mut prod, BEES);
    interleave(&mut prod);
    riffle(&mut save);

    quote! {
        unsafe {
            #(#load)*
            for _ in 0..p {
                #(#yvecs)*
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
