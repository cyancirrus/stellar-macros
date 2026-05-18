#![allow(unused)]
use crate::avx2::emitters;
use crate::instructs::perms::{interleave_partitions, riffle, riffle_partitions};
use proc_macro;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Result, Token, parse_macro_input};

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
impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let args = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
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
pub fn mult_alligned(input: proc_macro::TokenStream, i:usize, k:usize) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m: _, p, n: _, s_x, s_y, s_t} = args;
    let tids = emitters::name_tvecs(i);
    let yids = emitters::name_yvecs(k);
    let tvecs = emitters::load_vecs(&tids, &tptr, &s_t, i);
    let yvecs = emitters::load_vecs(&yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let prod = emitters::fma_product(&tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::handle_tail(&tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k);
    let save = emitters::write_outcome(&tids, &tptr, &s_t, i);
    
    // riffle(&mut tvecs);
    // riffle_partitions(&mut prod, k);
    // interleave_partitions(&mut prod, k);
    // riffle(&mut save);

    quote! {
        unsafe {
            #(#tvecs)*
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
                #yinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
#[rustfmt::skip]
pub fn mult_unalligned(input: proc_macro::TokenStream, i:usize, k:usize) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs { xptr, yptr, tptr, m, p, n, s_x, s_y, s_t} = args;
    let tids = emitters::name_tvecs(i);
    let yids = emitters::name_yvecs(k);
    let (mask_m, mask_n) = emitters::name_masks();
    // instructs
    let masks = emitters::load_masks(&m, &n); 
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let prod = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(&mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k);
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    
    // riffle(&mut tvecs);
    // riffle_partitions(&mut prod, k);
    // interleave_partitions(&mut prod, k);
    // riffle(&mut save);

    quote! {
        unsafe {
            #masks
            #(#tvecs)*
            for _ in 0..#p / #k {
                #(#yvecs)*
                #(#prod)*
                #yinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
use proc_macro2::{Ident, TokenStream};
pub type Vars = Vec<Ident>;
pub type Instr = Vec<TokenStream>;

fn threshold(threshold: &Ident, m: &Expr, p: &Expr) -> proc_macro::TokenStream {
    quote! {
        let #threshold = #m.min(#p);
    }
    .into()
}
pub fn mult_lower_tri(
    input: proc_macro::TokenStream,
    i: usize,
    k: usize,
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs {
        xptr,
        yptr,
        tptr,
        m,
        p,
        n,
        s_x,
        s_y,
        s_t,
    } = args;
    let tids = emitters::name_tvecs(i);
    let yids = emitters::name_yvecs(k);
    let hid = emitters::name_threshold();
    let (mask_m, mask_n) = emitters::name_masks();
    // instructs
    let masks = emitters::load_masks(&m, &n);
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let tri = emitters::handle_ltri(
        &mask_n, &tids, &xptr, &yptr, &s_x, &s_y, &yids[0], i);
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #masks
            #(#tvecs)*
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #yinc
            }
            #(#tail)*
            #(#tri)*
            #(#save)*
        }
    }
    .into()
}
pub fn mult_upper_tri(
    input: proc_macro::TokenStream,
    i: usize,
    k: usize,
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as KernelArgs);
    let KernelArgs {
        xptr,
        yptr,
        tptr,
        m,
        p,
        n,
        s_x,
        s_y,
        s_t,
    } = args;
    let tids = emitters::name_tvecs(i);
    let yids = emitters::name_yvecs(k);
    let hid = emitters::name_threshold();
    let (mask_m, mask_n) = emitters::name_masks();
    // instructs
    let masks = emitters::load_masks(&m, &n);
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let tri = emitters::handle_utri( &mask_n, &tids, &xptr, &yptr, &s_x, &s_y, &yids[0], i);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #masks
            #(#tvecs)*
            #(#tri)*
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #yinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
