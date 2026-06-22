use proc_macro::TokenStream;
mod avx2;
mod instructs;

const M: usize = 8;
const B: usize = 4;

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
pub fn kernel_lmult_lower_tri(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::lmult_lower_tri(input, M, B)
}
#[proc_macro]
pub fn kernel_lmult_upper_tri(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::lmult_upper_tri(input, M, B)
}
#[proc_macro]
pub fn kernel_mult_alligned(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::mult_alligned(input, M, B)
}
#[proc_macro]
pub fn kernel_mult_unalligned(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::mult_unalligned(input, M, B)
}
// #[proc_macro]
// pub fn kernel_tmult_alligned(input: TokenStream) -> TokenStream {
//     avx2::kernel::matmul::tmult_alligned(input, M, B)
// }
#[proc_macro]
pub fn kernel_tmult_unalligned(input: TokenStream) -> TokenStream {
    avx2::kernel::matmul::tmult_unalligned(input, M, B)
}
