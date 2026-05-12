use proc_macro::TokenStream;
mod avx2;
mod instructs;

#[proc_macro]
pub fn avx2_pack_simd_line(input: TokenStream) -> TokenStream {
    avx2::pack::pack_simd_line(input)
}

#[proc_macro]
pub fn avx2_pack_simd_line_alligned(input: TokenStream) -> TokenStream {
    avx2::pack::pack_simd_line_alligned(input)
}

#[proc_macro]
pub fn avx2_pack_simd_line_unalligned(input: TokenStream) -> TokenStream {
    avx2::pack::pack_simd_line_unalligned(input)
}

#[proc_macro]
pub fn kernel_mult_alligned(input: TokenStream) -> TokenStream {
    avx2::kernel::alligned::mult_alligned(input)
}
