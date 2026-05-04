use proc_macro2;

pub fn parse_usize(tt: &proc_macro2::TokenTree) -> usize {
    if let proc_macro2::TokenTree::Literal(lit) = tt {
        lit.to_string().parse().unwrap()
    } else {
        panic!("usize parsing failure");
    }
}
