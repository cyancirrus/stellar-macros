use quote::{format_ident, quote};
use syn::Expr;
use proc_macro2::{Ident, TokenStream};

pub type Vars = Vec<Ident>;
pub type Instr = Vec<TokenStream>;

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
    (
        format_ident!("mask_m"),
        format_ident!("mask_n"),
    )
}
pub fn load_vecs(vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    let mut loads = Vec::with_capacity(card);
    for (idx, name) in vars.iter().enumerate() {
        loads.push(quote! {
            let mut #name = _mm256_loadu_ps(#ptr.add(#idx * #stride));
        });
    }
    loads
}
pub fn mload_vecs(mask:&Ident, vars: &Vars, ptr: &Expr, stride: &Expr, card: usize) -> Instr {
    let mut loads = Vec::with_capacity(card);
    for (idx, name) in vars.iter().enumerate() {
        loads.push(quote! {
            let mut #name = mask_load(#ptr.add(#idx * #stride), #mask);
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
pub fn write_outcome(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut saves = Vec::with_capacity(m);
    for (idx, ident) in tids.iter().enumerate() {
        saves.push(quote! {
            _mm256_storeu_ps(#tptr.add(#idx * #s_t), #ident);
        });
    }
    saves
}
pub fn mwrite_outcome(mask_m:&Ident, mask_n:&Ident, tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut saves = Vec::with_capacity(m);
    for (idx, var) in tids.iter().enumerate() {
        saves.push(quote! {
            mask_store_ctrl(#tptr.add(#idx * #s_t), #mask_n, #var, #mask_m[#idx]);
        });
    }
    saves
}
pub fn fma_product(tids: &Vars, yids: &Vars, xptr: &Expr, s_x: &Expr, m: usize, k: usize) -> Instr {
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
pub fn mfma_product(mask:&Ident, tids: &Vars, yids: &Vars, xptr: &Expr, s_x: &Expr, m: usize, k: usize) -> Instr {
    let mut products = Vec::with_capacity(m * k);
    for (bdx, bname) in yids.iter().enumerate() {
        for (idx, ident) in tids.iter().enumerate() {
            products.push(quote! {
                fma_gated!(#ident, #xptr.add(#idx * #s_x + #bdx), #mask[#idx], #bname);
            });
        }
    }
    products
}
