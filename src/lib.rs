use proc_macro::TokenStream;
mod avx2;
mod instructs;

const M: usize = 8;
const B: usize = 2;

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
    avx2::kernel::matmul::mult_alligned(input, B, M)
}
#[proc_macro]
pub fn kernel_mult_unalligned(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::mult_unalligned(input, B, M)
}
