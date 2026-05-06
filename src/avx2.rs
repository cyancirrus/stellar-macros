use quote::quote;
use {proc_macro};
use syn::{parse::{Parse, ParseStream}, punctuated::Punctuated};
use syn::{Result, Token, Expr, parse_macro_input};

const AVX2_SIMD_WIDTH: usize = 8;
// const BLOCK_WIDTH: usize = 256;
const BLOCK_WIDTH: usize = 4;
/// # pack_simd_line transfers a copy of data from d to pack
/// * to inverse simply exchange d and b
/// - d ~ M(r, s)
///
/// * dptr: contains the source data of x sliced to begin at mc
/// * bptr: contains the target pack for the outer iteration loop
/// * block: stride of block
/// * source: columns from source
struct PackSimdArgs{
    bptr: Expr,
    dptr: Expr,
}

struct PackUSimdArgs {
    bptr: Expr,
    dptr: Expr,
    source: Expr,
}

impl Parse for PackSimdArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let vars = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
        let mut iter = vars.into_iter();
        let bptr: Expr = iter.next().ok_or(input.error("failed to parse bptr"))?;
        let dptr: Expr = iter.next().ok_or(input.error("failed to parse dptr"))?;
        Ok(Self { bptr, dptr })
    }
}

impl Parse for PackUSimdArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let vars = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
        let mut iter = vars.into_iter();
        let bptr: Expr = iter.next().ok_or(input.error("failed to parse bptr"))?;
        let dptr: Expr = iter.next().ok_or(input.error("failed to parse dptr"))?;
        let source: Expr = iter.next().ok_or(input.error("failed to parse source"))?;
        Ok(Self { bptr, dptr, source})
    }
}

// pub fn pack_simd_line_alligned(
//     input: proc_macro::TokenStream
// // ) -> proc_macro::TokenStream {
// ) -> proc_macro::TokenStream {
//     let unroll_factor = 4;
//     let args = parse_macro_input!(input as PackSimdArgs);
//     let PackSimdArgs { bptr, dptr } = args;
//     let mut unroll = Vec::new();
//     // 4 unroll 
//     for o in (0..unroll_factor * AVX2_SIMD_WIDTH).step_by(AVX2_SIMD_WIDTH) {
//         unroll.push(quote! {
//             _mm256_storeu_ps(
//                 #bptr.add(i + #o),
//                 _mm256_loadu_ps(#dptr.add(i + #o))
//             );
//         });
//     }
//     // loop over the 4 unroll
//     let stride = unroll_factor * AVX2_SIMD_WIDTH;
//     quote! {
//         for i in (0..#AVX2_SIMD_WIDTH).step_by(#stride) {
//             #(#unroll)*
//         }
//     }.into()
// }

pub fn pack_simd_line_alligned(
    input: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as PackSimdArgs);
    let PackSimdArgs { bptr, dptr } = args;
    let mut tokens = Vec::new();
    for o in (0..BLOCK_WIDTH).step_by(AVX2_SIMD_WIDTH) {
        tokens.push(quote! {
            _mm256_storeu_ps(
                #bptr.add(#o),
                _mm256_loadu_ps(#dptr.add(#o))
            );
        });
    }
    quote! {#(#tokens)*}.into()
}
pub fn pack_simd_line_unalligned(
    input: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as PackUSimdArgs);
    let PackUSimdArgs { bptr, dptr, source } = args;
    let mut tokens = Vec::new();
    
    // x & 7 == x % 8;
    for o in (0..BLOCK_WIDTH).step_by(AVX2_SIMD_WIDTH) {
        tokens.push(
            quote! {
                _mm256_storeu_ps(
                    #bptr.add(#o),
                    _mm256_maskload_ps(
                        #dptr.add(#o),
                        _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i)
                    )
                );
            }
            // quote! {
            //     _mm256_maskstore_ps(
            //         #bptr.add(#o),
            //         _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i),
            //         _mm256_maskload_ps(
            //             #dptr.add(#o),
            //             _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i)
            //         )
            //     );
            // }
        );
    }
    quote! {#(#tokens)*}.into()
}
// pub fn pack_simd_line_unalligned(
//     input: proc_macro::TokenStream
// ) -> proc_macro::TokenStream {
//     let args = parse_macro_input!(input as PackUSimdArgs);
//     let PackUSimdArgs { bptr, dptr, source } = args;
//     let mut tokens = Vec::new();
    
//     // x & 7 == x % 8;
//     for o in (0..BLOCK_WIDTH).step_by(AVX2_SIMD_WIDTH) {
//         tokens.push(
//             quote! {
//                 _mm256_maskstore_ps(
//                     #bptr.add(#o),
//                     _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i),
//                     _mm256_maskload_ps(
//                         #dptr.add(#o),
//                         _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i)
//                     )
//                 );
//             }
//         );
//     }
//     quote! {#(#tokens)*}.into()
// }
