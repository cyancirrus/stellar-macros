use quote::{format_ident, quote};
use syn::Expr;

pub type Vars = Vec<proc_macro2::Ident>;
pub type Instr = Vec<proc_macro2::TokenStream>;

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
pub fn load_tvecs(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut loads = Vec::with_capacity(m);
    for (idx, ident) in tids.iter().enumerate() {
        loads.push(quote! {
            let mut #ident = _mm256_loadu_ps(#tptr.add(#idx * #s_t));
        });
    }
    loads
}
pub fn mload_tvecs(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut loads = Vec::with_capacity(m);
    for (idx, ident) in tids.iter().enumerate() {
        loads.push(quote! {
            let mut #ident = mask_load(#tptr.add(#idx * #s_t), mask_n);
        });
    }
    loads
}
pub fn load_yvecs(yids: &Vars, yptr: &Expr, s_y: &Expr, k: usize) -> Instr {
    let mut loads = Vec::with_capacity(k);
    for (bdx, bee) in yids.iter().enumerate() {
        loads.push(quote! {
            let #bee = _mm256_loadu_ps(#yptr + #bdx * #s_y);
        });
    }
    loads
}
pub fn mload_yvecs(yids: &Vars, yptr: &Expr, s_y: &Expr, k: usize) -> Instr {
    let mut loads = Vec::with_capacity(k);
    for (bdx, bee) in yids.iter().enumerate() {
        loads.push(quote! {
            let #bee = mask_load(#yptr + #bdx * #s_y, mask_n);
        });
    }
    loads
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
pub fn mwrite_outcome(tids: &Vars, tptr: &Expr, s_t: &Expr, m: usize) -> Instr {
    let mut saves = Vec::with_capacity(m);
    for (idx, var) in tids.iter().enumerate() {
        saves.push(quote! {
            mask_store_ctrl(#tptr.add(#idx * #s_t), mask_n, #var, mask_m[#idx]);
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
pub fn mfma_product(tids: &Vars, yids: &Vars, xptr: &Expr, s_x: &Expr, m: usize, k: usize) -> Instr {
    let mut products = Vec::with_capacity(m * k);
    for (bdx, bname) in yids.iter().enumerate() {
        for (idx, ident) in tids.iter().enumerate() {
            products.push(quote! {
                fma_gated!(#ident, #xptr.add(#idx * #s_x + #bdx), mask_m[#idx], #bname);
            });
        }
    }
    products
}
