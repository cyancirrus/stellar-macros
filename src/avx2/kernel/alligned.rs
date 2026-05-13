#![allow(unused)]
use crate::avx2::common;
use crate::avx2::common::{Instr, Vars, index_matrix};
use crate::instructs::perms::{interleave, interleave_partitions, riffle, riffle_partitions};
use proc_macro;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Result, Token, parse_macro_input};

const M: usize = 8;
const B: usize = 2;

macro_rules! parse_next {
    ($args:expr, $input:expr) => {
        $args.next().ok_or($input.error("variable not found"))?
    };
}
use proc_macro2::{Ident, TokenStream};


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
#[rustfmt::skip]
fn mult_kernel(input: proc_macro::TokenStream, i:usize, k:usize) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = common::name_tvecs(i);
    let yids = common::name_yvecs(k);
    let mut tvecs = common::load_vecs(&tids, &tptr, &s_t, i);
    let mut yvecs = common::load_vecs(&yids, &yptr, &s_y, k);
    let mut prod = common::fma_product(&tids, &yids, &xptr, &s_x, i, k);
    let mut save = common::write_outcome(&tids, &tptr, &s_t, i);
    
    riffle(&mut tvecs);
    riffle_partitions(&mut prod, k);
    interleave_partitions(&mut prod, k);
    riffle(&mut save);

    quote! {
        unsafe {
            #(#tvecs)*
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
            }
            #(#save)*
        }
    }
    .into()
}

#[rustfmt::skip]
pub fn mult_alligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = common::name_tvecs(M);
    let yids = common::name_yvecs(B);
    let mut yvecs = common::load_vecs(&yids, &yptr, &s_y, B);
    let mut tvecs = common::load_vecs(&tids, &tptr, &s_t, M);
    let mut prod = common::fma_product(&tids, &yids, &xptr, &s_x, M, B);
    let mut save = common::write_outcome(&tids, &tptr, &s_t, M);
    
    riffle(&mut tvecs);
    riffle_partitions(&mut prod, B);
    interleave_partitions(&mut prod, B);
    riffle(&mut save);

    quote! {
        unsafe {
            #(#tvecs)*
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
            }
            #(#save)*
        }
    }
    .into()
}
#[rustfmt::skip]
pub fn mult_unalligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = common::name_tvecs(M);
    let yids = common::name_yvecs(B);
    let (mask_m, mask_n) = common::name_masks();
    // instructs
    let masks = common::load_masks(&m, &n); 
    let mut tvecs = common::mload_vecs(&mask_m, &tids, &tptr, &s_t, M);
    let mut yvecs = common::mload_vecs(&mask_n, &yids, &yptr, &s_y, B);
    let mut prod = common::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, M, B);
    let mut save = common::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, M);
    
    riffle(&mut tvecs);
    riffle_partitions(&mut prod, B);
    interleave_partitions(&mut prod, B);
    riffle(&mut save);

    quote! {
        unsafe {
            #masks
            #(#tvecs)*
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
            }
            #(#save)*
        }
    }
    .into()
}
