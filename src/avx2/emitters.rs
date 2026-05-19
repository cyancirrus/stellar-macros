#![allow(unused)]
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::Expr;

pub type Vars = Vec<Ident>;
pub type Args = Vec<Expr>;
pub type Instr = Vec<TokenStream>;

pub fn index_matrix(ptr: &Expr, stride: &Expr, row: usize, col: usize) -> TokenStream {
    match (row, col) {
        (0, 0) => quote! { #ptr },
        (0, c) => quote! { #ptr.add(#c) },
        (r, 0) => quote! { #ptr.add(#stride * #r) },
        (r, c) => quote! { #ptr.add(#stride * #r + #c) },
    }
}
pub fn name_range(prefix: &str, m: usize) -> Vars {
    let mut idents = Vec::with_capacity(m);
    for idx in 0..m {
        idents.push(format_ident!("{prefix:?}{idx:?}"));
    }
    idents
}
pub fn name(content: &str) -> Ident {
    format_ident!("{content:?}")
}
pub fn increment(ptr: &Expr, stride: &Expr, row: usize, col: usize) -> TokenStream {
    let addr = index_matrix(&ptr, &stride, row, col);
    quote! {
        #ptr = #addr;
    }
}
pub fn threshold(threshold: &Ident, m: &Expr, p: &Expr) -> TokenStream {
    quote! {
        let #threshold = #m.min(#p);
    }
}
pub fn init_var(name: &Ident, val: &TokenStream) -> TokenStream {
    quote! {
        let mut #name = #val
    }
}
pub fn lvec(name: &Ident, ptr: &Expr, stride: &Expr, row: usize, col: usize) -> TokenStream {
    let addr = index_matrix(&ptr, &stride, row, 0);
    quote! {
        let mut #name = _mm256_loadu_ps(#addr);
    }
}
pub fn mlvec(
    mask: &Ident,
    name: &Ident,
    ptr: &Expr,
    stride: &Expr,
    row: usize,
    col: usize,
) -> TokenStream {
    let addr = index_matrix(&ptr, &stride, row, col);
    quote! {
        let mut #name = mask_load(#mask, #addr);
    }
}
pub fn fma(
    name: &Ident,
    b: &Expr,
    ptr: &Expr,
    stride: &Expr,
    row: usize,
    col: usize,
) -> TokenStream {
    let addr = index_matrix(&ptr, &stride, row, col);
    quote! {
        fma_accum!(#name, #addr, #b);
    }
}
#[rustfmt::skip]
pub fn load_vecs(vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    vars.iter().enumerate().map(
        |(idx, name)| lvec(&name, &ptr, &stride, idx,0)
    ).collect()
}
#[rustfmt::skip]
pub fn mload_vecs(mask: &Ident, vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    vars.iter().enumerate().map(
        |(idx, name)| mlvec(&mask,&name, &ptr, &stride, idx, 0)
    ).collect()
}

pub fn fma_product(tids: &Vars, yids: &Vars, xptr: &Expr, s_x: &Expr, m: usize, k: usize) -> Instr {
    let mut products = Vec::with_capacity(m * k);
    for (bdx, b) in yids.iter().enumerate() {
        for (idx, ident) in tids.iter().enumerate() {
            let addr = index_matrix(&xptr, &s_x, idx, bdx);
            products.push(quote! {
                fma_accum!(#ident, #addr, #b);
            });
        }
    }
    products
}
pub fn mfma_product(
    mask: &Ident,
    tids: &Vars,
    yids: &Vars,
    xptr: &Expr,
    s_x: &Expr,
    i: usize,
    k: usize,
) -> Instr {
    let mut products = Vec::with_capacity(i * k);
    for (bdx, bname) in yids.iter().enumerate() {
        for (idx, ident) in tids.iter().enumerate() {
            let addr = index_matrix(&xptr, &s_x, idx, bdx);
            products.push(quote! {
                mfma_accum!(#mask[#idx], #ident, #addr, #bname);
            });
        }
    }
    products
}
pub fn write_outcome(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut saves = Vec::with_capacity(m);
    for (idx, ident) in tids.iter().enumerate() {
        let addr = index_matrix(&tptr, &s_t, idx, 0);
        saves.push(quote! {
            _mm256_storeu_ps(#addr, #ident);
        });
    }
    saves
}
pub fn mwrite_outcome(
    mask_m: &Ident,
    mask_n: &Ident,
    tids: &Vars,
    tptr: &Expr,
    s_t: &Expr,
    m: usize,
) -> Instr {
    let mut saves = Vec::with_capacity(m);
    for (idx, var) in tids.iter().enumerate() {
        let addr = index_matrix(&tptr, &s_t, idx, 0);
        saves.push(quote! {
            mask_store_ctrl(#mask_m[#idx], #mask_n, #addr, #var);
        });
    }
    saves
}
fn initialize_q(k: usize) -> usize {
    if k.count_ones() == 1 {
        k >> 1
    } else {
        1 << (usize::BITS - k.leading_zeros() - 1)
    }
}
/// handle_tail
///
/// * when unrolling b terms we need to handle the tail
//  k := static unwrap
//  p := runtime variable
pub fn handle_tail(
    tids: &Vars,
    yids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    p: &Expr,
    k: usize,
) -> Instr {
    // binary decomp of the k variable
    let mut q = initialize_q(k);
    let mut tail = Vec::new();
    let yname = format_ident!("yptr");
    while q > 0 {
        let mut section = Vec::new();
        for bdx in 0..q {
            let bname = &yids[bdx];
            let yaddr = index_matrix(&yptr, &s_y, bdx, 0);
            section.push(quote! {
                let #bname = _mm256_loadu(#yaddr);
            });
        }

        for bdx in 0..q {
            let bname = &yids[bdx];
            for (idx, ident) in tids.iter().enumerate() {
                let addr = index_matrix(&xptr, &s_x, idx, bdx);
                section.push(quote! {
                    fma_accum!(#ident, #addr, #bname);
                });
            }
        }
        let naddr = index_matrix(&yptr, &s_y, q, 0);
        tail.push(quote! {
            if #q & #p != 0 {
                #(#section)*
                #yname = #naddr;
            }
        });
        q >>= 1
    }
    tail
}
pub fn mhandle_tail(
    mask_m: &Ident,
    mask_n: &Ident,
    tids: &Vars,
    yids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    p: &Expr,
    k: usize,
) -> Instr {
    let mut q = initialize_q(k);
    let mut tail = Vec::new();
    let yname = format_ident!("yptr");
    while q > 0 {
        let mut section = Vec::new();
        for bdx in 0..q {
            let bname = &yids[bdx];
            let yaddr = index_matrix(&yptr, &s_y, bdx, 0);
            section.push(quote! {
                let #bname = mask_load(#mask_n, #yaddr);
            });
        }
        for bdx in 0..q {
            let bname = &yids[bdx];
            for (idx, ident) in tids.iter().enumerate() {
                let addr = index_matrix(&xptr, &s_x, idx, bdx);
                section.push(quote! {
                    mfma_accum!(#mask_m[#idx], #ident, #addr, #bname);
                });
            }
        }
        let naddr = index_matrix(&yptr, &s_y, q, 0);
        tail.push(quote! {
            if #q & #p != 0 {
                #(#section)*
                #yname = #naddr;
            }
        });
        q >>= 1
    }
    tail
}
pub fn lhandle_lowtri(
    mask_n: &Ident,
    tids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    b: &Ident,
    m: usize,
) -> Instr {
    let mut tri = Vec::new();
    for i in 0..m {
        let mut fmas = Vec::new();
        for idx in i..m {
            let ident = &tids[idx];
            let addr = index_matrix(&xptr, &s_x, idx, 0);
            fmas.push(quote! {
                fma_accum!(#ident, #addr, #b);
            });
        }
        let nyaddr = index_matrix(&yptr, &s_y, 1, 0);
        let nxaddr = index_matrix(&xptr, &s_x, 0, 1);
        tri.push(quote! {
            {
                let #b = mask_load(#mask_n, #yptr);
                #(#fmas)*
                #xptr = #nxaddr;
                #yptr = #nyaddr;
            }
        });
    }
    tri
}
pub fn lhandle_uptri(
    mask_n: &Ident,
    tids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    b: &Ident,
    m: usize,
) -> Instr {
    let mut tri = Vec::new();
    for i in 0..m {
        let mut fmas = Vec::new();
        for idx in 0..=i {
            let ident = &tids[idx];
            let addr = index_matrix(&xptr, &s_x, idx, 0);
            fmas.push(quote! {
                fma_accum!(#ident, #addr, #b);
            });
        }
        let nyaddr = index_matrix(&yptr, &s_y, 1, 0);
        let nxaddr = index_matrix(&xptr, &s_x, 0, 1);
        tri.push(quote! {
            {
                let #b = mask_load(#mask_n, #yptr);
                #(#fmas)*
                #xptr = #nxaddr;
                #yptr = #nyaddr;
            }
        });
    }
    tri
}
/// rhandle_lowtrie
///
/// A * L
/// handles the top part of L column slice
pub fn rhandle_lowtrie(
    mask_n: &Ident,
    mask_t: &Ident,
    tids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    h: &Ident,
    b: &Ident,
) -> TokenStream {
    let mut fmas = Vec::new();
    for (idx, ident) in tids.iter().enumerate() {
        let xaddr = index_matrix(&xptr, &s_x, idx, 0);
        fmas.push(quote! {
            mfma_accum!(#mask_t[#idx], #ident, #xaddr, #b);
        });
    }
    let ynaddr = index_matrix(&yptr, &s_y, 1, 0);
    let xnaddr = index_matrix(&xptr, &s_x, 0, 1);
    quote! {
        for i in 0..#h {
            #mask_t[i] = #mask_n[i];
            let #b = mask_load(#mask_t, #yptr);
            #(#fmas)*
            #yptr = #ynaddr;
            #xptr = #xnaddr;
        }
    }
}
/// rhandle_uptrie
///
/// A * U
/// handles the top part of U row slice
pub fn rhandle_uptrie(
    mask_n: &Ident,
    mask_t: &Ident,
    tids: &Vars,
    xptr: &Expr,
    yptr: &Expr,
    s_x: &Expr,
    s_y: &Expr,
    h: &Ident,
    b: &Ident,
) -> TokenStream {
    let mut fmas = Vec::new();
    for (idx, ident) in tids.iter().enumerate() {
        let xaddr = index_matrix(&xptr, &s_x, idx, 0);
        fmas.push(quote! {
            mfma_accum!(#mask_t[#idx], #ident, #xaddr, #b);
        });
    }
    let ynaddr = index_matrix(&yptr, &s_y, 1, 0);
    let xnaddr = index_matrix(&xptr, &s_x, 0, 1);
    quote! {
        for j in (0..#h).rev() {
            let #b = mask_load(#mask_t, #yptr);
            #(#fmas)*
            #yptr = #ynaddr;
            #xptr = #xnaddr;
            #mask_t[j] = 0;
        }
    }
}
