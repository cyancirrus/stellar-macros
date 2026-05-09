use proc_macro::TokenStream;
mod avx2;

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
