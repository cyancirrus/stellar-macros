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
    let mut idents = Vec::with_capacity(M);
    for idx in 0..M {
        idents.push(format_ident!("r{idx:?}"));
    }
    idents
}
fn bees() -> Vars {
    let mut bees = Vec::with_capacity(BEES);
    for idx in 0..BEES {
        bees.push(format_ident!("b{idx:?}"));
    }
    bees
}
fn load_target(irows: &Vars, tptr: &Expr, s_t: &Expr) -> Instr {
    let mut loads = Vec::with_capacity(M);
    for (idx, ident) in irows.iter().enumerate() {
        loads.push(quote! {
            let mut #ident = _mm256_loadu_ps(#tptr.add(#idx * #s_t));
        });
    }
    loads
}
fn load_yvecs(bees: &Vars, yptr: &Expr, s_y: &Expr) -> Instr {
    let mut loads = Vec::with_capacity(BEES);
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
pub fn mult_alligned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let rows = irows();
    let bees = bees();
    let mut yvecs = load_yvecs(&bees, &yptr, &s_y);
    let mut load = load_target(&rows, &tptr, &s_t);
    let mut prod = fma_product(&rows, &bees, &xptr, &s_x);
    let mut save = write_outcome(&rows, &tptr, &s_t);
    
    riffle(&mut load);
    riffle_partitions(&mut prod, BEES);
    interleave_partitions(&mut prod, BEES);
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
    riffle_partitions(&mut prod, BEES);
    interleave_partitions(&mut prod, BEES);
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
