#![allow(unused)]
use crate::avx2::common;
use crate::avx2::common::{Instr, Vars};
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
pub fn mult_alligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = common::name_tvecs(M);
    let yids = common::name_yvecs(B);
    let mut yvecs = common::load_yvecs(&yids, &yptr, &s_y, B);
    let mut load = common::load_tvecs(&tids, &tptr, &s_t, M);
    let mut prod = common::fma_product(&tids, &yids, &xptr, &s_x, M, B);
    let mut save = common::write_outcome(&tids, &tptr, &s_t, M);
    
    riffle(&mut load);
    riffle_partitions(&mut prod, B);
    interleave_partitions(&mut prod, B);
    riffle(&mut save);

    quote! {
        unsafe {
            #(#load)*
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
    let mut yvecs = common::mload_yvecs(&yids, &yptr, &s_y, B);
    let mut load = common::mload_tvecs(&tids, &tptr, &s_t, M);
    let mut prod = common::mfma_product(&tids, &yids, &xptr, &s_x, M, B);
    let mut save = common::mwrite_outcome(&tids, &tptr, &s_t, M);
    
    riffle(&mut load);
    riffle_partitions(&mut prod, B);
    interleave_partitions(&mut prod, B);
    riffle(&mut save);

    quote! {
        unsafe {
            let mask_m = MASK[#m];
            #(#load)*
            let mask_n = MASK[#n];
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
            }
            #(#save)*
        }
    }
    .into()
}
