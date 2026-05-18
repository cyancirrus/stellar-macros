use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::Expr;

pub type Vars = Vec<Ident>;
pub type Instr = Vec<TokenStream>;

pub fn index_matrix(ptr: &Expr, stride: &Expr, row: usize, col: usize) -> TokenStream {
    match (row, col) {
        (0, 0) => quote! { #ptr },
        (0, c) => quote! { #ptr.add(#c) },
        (r, 0) => quote! { #ptr.add(#stride * #r) },
        (r, c) => quote! { #ptr.add(#stride * #r + #c) },
    }
}

pub fn name_tvecs(m: usize) -> Vars {
    let mut idents = Vec::with_capacity(m);
    for idx in 0..m {
        idents.push(format_ident!("r{idx:?}"));
    }
    idents
}
pub fn name_yvecs(k: usize) -> Vars {
    let mut yids = Vec::with_capacity(k);
    for idx in 0..k {
        yids.push(format_ident!("b{idx:?}"));
    }
    yids
}
pub fn name_masks() -> (Ident, Ident) {
    (format_ident!("mask_m"), format_ident!("mask_n"))
}
pub fn name_threshold() -> Ident {
    format_ident!("threshold")
}
pub fn increment(ptr: &Expr, stride:&Expr, row:usize, col:usize) ->  TokenStream {
    let addr = index_matrix(&ptr, &stride, row, col);
    quote! {
        #ptr = #addr;
    }
}
pub fn load_vecs(vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    let mut loads = Vec::with_capacity(card);
    for (idx, name) in vars.iter().enumerate() {
        let addr = index_matrix(&ptr, &stride, idx, 0);
        loads.push(quote! {
            let mut #name = _mm256_loadu_ps(#addr);
        });
    }
    loads
}
pub fn mload_vecs(mask: &Ident, vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    let mut loads = Vec::with_capacity(card);
    for (idx, name) in vars.iter().enumerate() {
        let addr = index_matrix(&ptr, &stride, idx, 0);
        loads.push(quote! {
            let mut #name = mask_load(#addr, #mask);
        });
    }
    loads
}
pub fn load_masks(m: &Expr, n: &Expr) -> TokenStream {
    quote! {
        let mask_m = MASK[#m];
        let mask_n = MASK[#n];
    }
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
            mask_store_ctrl(#addr, #mask_n, #var, #mask_m[#idx]);
        });
    }
    saves
}

/// handle_tail
///
/// * when unrolling b terms we need to handle the tail
//  k := static unwrap
//  p := runtime variable
fn initialize_q(k: usize) -> usize {
    if k.count_ones() == 1 {
        k >> 1
    } else {
        1 << (usize::BITS - k.leading_zeros() - 1)
    }
}
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
pub fn handle_ltri(mask_n:&Ident, tids:&Vars, xptr:&Expr, yptr:&Expr, s_x:&Expr, s_y:&Expr, b:&Ident, m: usize) -> Instr {
    let mut tri = Vec::new();
    for i in 0..m {
        let mut fmas = Vec::new();
        for idx in i..m {
            let ident = &tids[idx];
            let addr = index_matrix(&xptr, &s_x, idx, 0);
            fmas.push(quote! {
                fma_accum!(#ident, #addr, #b)
            });
        }
        let nyaddr = index_matrix(&yptr, &s_y, 1, 0);
        let nxaddr = index_matrix(&xptr, &s_x, 0, 1);
        tri.push(quote! {
            {
                let #b = mask_load(#mask_n, #yptr);
                #(#fmas)*
                #xptr = #nxaddr
                #yptr = #nyaddr
            }
        });
    }
    tri
}
pub fn handle_utri(mask_n:&Ident, tids:&Vars, xptr:&Expr, yptr:&Expr, s_x:&Expr, s_y:&Expr, b:&Ident, m: usize) -> Instr {
    let mut tri = Vec::new();
    for i in 0..m {
        let mut fmas = Vec::new();
        for idx in i..m {
            let ident = &tids[idx];
            let addr = index_matrix(&xptr, &s_x, idx, 0);
            fmas.push(quote! {
                fma_accum!(#ident, #addr, #b)
            });
        }
        let nyaddr = index_matrix(&yptr, &s_y, 1, 0);
        let nxaddr = index_matrix(&xptr, &s_x, 0, 1);
        tri.push(quote! {
            {
                let #b = mask_load(#mask_n, #yptr);
                #(#fmas)*
                #xptr = #nxaddr
                #yptr = #nyaddr
            }
        });
    }
    tri
}
