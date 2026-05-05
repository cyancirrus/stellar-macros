use crate::parse;
use quote::quote;
use {proc_macro, proc_macro2};

const AVX2_SIMD_WIDTH: usize = 8;

/// # pack_simd_line transfers a copy of data from d to pack
/// * to inverse simply exchange d and b
/// - d ~ M(r, s)
///
/// * dptr: contains the source data of x sliced to begin at mc
/// * bptr: contains the target pack for the outer iteration loop
/// * block: stride of block
/// * source: columns from source
pub fn pack_simd_line_alligned(
    input: proc_macro::TokenStream
    // bptr: &proc_macro2::TokenTree,
    // dptr: &proc_macro2::TokenTree,
    // boffset: &proc_macro2::TokenTree,
    // doffset: &proc_macro2::TokenTree,
    // block: usize,
) -> proc_macro::TokenStream {
    let args = proc_macro2::TokenStream::from(input);
    let args: Vec<proc_macro2::TokenTree> = args.into_iter().collect();
    let bptr = &args[0];
    let dptr = &args[2];
    let boffset = &args[4];
    let doffset = &args[6];
    // source block size
    // let source = &args[8];
    // destination block size
    // let block = parse::parse_usize(&args[10]);
    let block:usize = 256;
    let mut tokens = Vec::new();
    for o in (0..block).step_by(AVX2_SIMD_WIDTH) {
        tokens.push(quote! {
            _mm256_storeu_ps(
                #bptr.add(#boffset + #o),
                _mm256_loadu_ps(#dptr.add(#doffset + #o))
            );
        });
    }
    quote! {#(#tokens)*}.into()
}
pub fn pack_simd_line_unalligned(
    input: proc_macro::TokenStream
    // bptr: &proc_macro2::TokenTree,
    // dptr: &proc_macro2::TokenTree,
    // boffset: &proc_macro2::TokenTree,
    // doffset: &proc_macro2::TokenTree,
    // block: usize,
    // source: &proc_macro2::TokenTree,
) -> proc_macro::TokenStream {
    let args = proc_macro2::TokenStream::from(input);
    let args: Vec<proc_macro2::TokenTree> = args.into_iter().collect();
    let bptr = &args[0];
    let dptr = &args[2];
    let boffset = &args[4];
    let doffset = &args[6];
    // source block size
    let source = &args[8];
    // destination block size
    // let block = parse::parse_usize(&args[10]);
    let block:usize = 256;
    let mut tokens = Vec::new();
    // x & 7 == x % 8;
    for o in (0..block).step_by(AVX2_SIMD_WIDTH) {
        tokens.push(
            quote! {
                _mm256_maskstore_ps(
                    #bptr.add(#boffset + #o),
                    _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i),
                    _mm256_maskload_ps(
                        #dptr.add(#doffset + #o),
                        _mm256_loadu_si256(MASK[#source.saturating_sub(#o).min(8)].as_ptr() as *const __m256i)
                    )
                );
            }
        );
    }
    quote! {#(#tokens)*}.into()
}
