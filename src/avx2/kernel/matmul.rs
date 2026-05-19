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
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);

    let tvecs = emitters::load_vecs(&tids, &tptr, &s_t, i);
    let yvecs = emitters::load_vecs(&yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&yptr, &s_y, 0, k);
    let prod = emitters::fma_product(&tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::handle_tail(&tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k);
    let save = emitters::write_outcome(&tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #(#tvecs)*
            for _ in 0..#p {
                #(#yvecs)*
                #(#prod)*
                #xinc
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
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);
    let mask_m = emitters::name("mask_m");
    let mask_n = emitters::name("mask_n");
    // instructs
    let load_mask_m = emitters::init_var(&mask_m, &quote! { MASK[#m] }); 
    let load_mask_n = emitters::init_var(&mask_m, &quote! { MASK[#n] }); 
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&yptr, &s_y, 0, k);
    let prod = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(&mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k);
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);

    quote! {
        unsafe {
            #load_mask_m
            #load_mask_n
            #(#tvecs)*
            for _ in 0..#p / #k {
                #(#yvecs)*
                #(#prod)*
                #yinc
                #xinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
pub fn lmult_lower_tri(
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
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);
    let hid = emitters::name("threshold");
    let mask_m = emitters::name("mask_m");
    let mask_n = emitters::name("mask_n");
    // instructs
    let load_mask_m = emitters::init_var(&mask_m, &quote! { MASK[#m] });
    let load_mask_n = emitters::init_var(&mask_m, &quote! { MASK[#n] });
    let load_thresh = emitters::init_var(&hid, &quote! { #m.min(#p) });
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&yptr, &s_y, 0, k);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let tri = emitters::lhandle_lowtri(&mask_n, &tids, &xptr, &yptr, &s_x, &s_y, &yids[0], i);
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #load_mask_m
            #load_mask_n
            #(#tvecs)*
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #xinc
                #yinc
            }
            #(#tail)*
            #(#tri)*
            #(#save)*
        }
    }
    .into()
}
pub fn lmult_upper_tri(
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
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);
    let hid = emitters::name("threshold");
    let hval = emitters::threshold(&hid, &m, &p);
    let mask_m = emitters::name("mask_m");
    let mask_n = emitters::name("mask_n");
    // instructs
    let load_mask_m = emitters::init_var(&mask_m, &quote! { MASK[#m] });
    let load_mask_n = emitters::init_var(&mask_m, &quote! { MASK[#n] });
    let load_thresh = emitters::init_var(&hid, &quote! { #m.min(#p) });
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let tri = emitters::lhandle_uptri(&mask_n, &tids, &xptr, &yptr, &s_x, &s_y, &yids[0], i);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&xptr, &s_x, 0, k);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #load_mask_m
            #load_mask_n
            #(#tvecs)*
            #(#tri)*
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #yinc
                #xinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
pub fn rmult_lower_tri(
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
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);
    let hid = emitters::name("threshold");
    let mask_m = emitters::name("mask_m");
    let mask_n = emitters::name("mask_n");
    let mask_t = emitters::name("mask_t");
    // instructs
    let load_thresh = emitters::init_var(&hid, &quote! { #m.min(#p) });
    let load_mask_m = emitters::init_var(&mask_m, &quote! { MASK[#m] });
    let load_mask_n = emitters::init_var(&mask_m, &quote! { MASK[#n] });
    let lfilter = emitters::init_var(&mask_t, &quote! { [0;#m] });
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&xptr, &s_x, 0, k);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let tri = emitters::rhandle_lowtrie(
        &mask_n, &mask_t, &tids, &xptr, &yptr, &s_x, &s_y, &hid, &yids[0],
    );
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #load_mask_m
            #load_mask_n
            #(#tvecs)*
            #hid
            #lfilter
            #tri
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #xinc
                #yinc
            }
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
pub fn rmult_upper_tri(
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
    let mask_m = emitters::name("mask_m");
    let mask_n = emitters::name("mask_n");
    let mask_t = emitters::name("mask_t");
    let hid = emitters::name("threshold");
    let tids = emitters::name_range("m", i);
    let yids = emitters::name_range("b", k);

    // instructs
    let load_mask_m = emitters::init_var(&mask_m, &quote! { MASK[#m] });
    let load_mask_n = emitters::init_var(&mask_m, &quote! { MASK[#n] });
    let load_mask_f = emitters::init_var(&mask_t, &quote! { #mask_n });
    let load_thresh = emitters::init_var(&hid, &quote! { #m.min(#p) });
    let tvecs = emitters::mload_vecs(&mask_m, &tids, &tptr, &s_t, i);
    let yvecs = emitters::mload_vecs(&mask_n, &yids, &yptr, &s_y, k);
    let yinc = emitters::increment(&yptr, &s_y, k, 0);
    let xinc = emitters::increment(&xptr, &s_x, 0, k);
    let fma = emitters::mfma_product(&mask_m, &tids, &yids, &xptr, &s_x, i, k);
    let tail = emitters::mhandle_tail(
        &mask_m, &mask_n, &tids, &yids, &xptr, &yptr, &s_x, &s_y, &p, k,
    );
    let tri = emitters::rhandle_uptrie(
        &mask_n, &mask_t, &tids, &xptr, &yptr, &s_x, &s_y, &hid, &yids[0],
    );
    let save = emitters::mwrite_outcome(&mask_m, &mask_n, &tids, &tptr, &s_t, i);
    quote! {
        unsafe {
            #load_mask_m
            #load_mask_n
            #(#tvecs)*
            #hid
            for _ in #hid..#p / #k {
                #(#yvecs)*
                #(#fma)*
                #xinc
                #yinc
            }
            #load_mask_f
            #tri
            #(#tail)*
            #(#save)*
        }
    }
    .into()
}
