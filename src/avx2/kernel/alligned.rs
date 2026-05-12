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
fn name_tvecs(m: usize) -> Vars {
    let mut idents = Vec::with_capacity(m);
    for idx in 0..m {
        idents.push(format_ident!("r{idx:?}"));
    }
    idents
}
fn name_yvecs(k: usize) -> Vars {
    let mut yids = Vec::with_capacity(k);
    for idx in 0..k {
        yids.push(format_ident!("b{idx:?}"));
    }
    yids
}
fn load_tvecs(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut loads = Vec::with_capacity(m);
    for (idx, ident) in tids.iter().enumerate() {
        loads.push(quote! {
            let mut #ident = _mm256_loadu_ps(#tptr.add(#idx * #s_t));
        });
    }
    loads
}
fn load_yvecs(yids: &Vars, yptr: &Expr, s_y: &Expr, k: usize) -> Instr {
    let mut loads = Vec::with_capacity(k);
    for (bdx, bee) in yids.iter().enumerate() {
        loads.push(quote! {
            let #bee = _mm256_loadu_ps(#yptr + #bdx * #s_y);
        });
    }
    loads
}
fn write_outcome(tids: &Vars, tptr: &Expr, s_t: &Expr) -> Instr {
    let mut saves = Vec::with_capacity(M);
    for (idx, ident) in tids.iter().enumerate() {
        saves.push(quote! {
            _mm256_storeu_ps(#tptr.add(#idx * #s_t), #ident);
        });
    }
    saves
}
fn fma_product(tids: &Vars, yids: &Vars, xptr: &Expr, s_x: &Expr, m: usize, k: usize) -> Instr {
    let mut products = Vec::with_capacity(m * k);
    for (bdx, b) in yids.iter().enumerate() {
        for (idx, ident) in tids.iter().enumerate() {
            products.push(quote! {
                fma_accum!(#ident, #xptr.add(#idx * #s_x + #bdx), #b);
            });
        }
    }
    products
}
#[rustfmt::skip]
pub fn mult_alligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = name_tvecs(M);
    let yids = name_yvecs(B);
    let mut yvecs = load_yvecs(&yids, &yptr, &s_y, B);
    let mut load = load_tvecs(&tids, &tptr, &s_t, M);
    let mut prod = fma_product(&tids, &yids, &xptr, &s_x, M, B);
    let mut save = write_outcome(&tids, &tptr, &s_t);
    
    riffle(&mut load);
    riffle_partitions(&mut prod, B);
    interleave_partitions(&mut prod, B);
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
