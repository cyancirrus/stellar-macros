#![allow(unused)]
pub fn interleave<T>(data: &mut [T]) {
    let h = data.len() >> 1;
    for i in (1..h).step_by(2) {
        data.swap(i, i + h);
    }
}
pub fn riffle<T>(data: &mut [T]) {
    // this won't work b/c it will start overlapping things oddly
    let mut seen: u64 = 0;
    let m = data.len() - 1;
    for i in 1..m {
        let mut v = i;
        loop {
            let t = 2 * v % m;
            // already swapped
            if seen & (1 << t) != 0 {
                break;
            }
            // data in hand at i
            data.swap(i, t);
            seen |= 1 << t;
            v = t;
        }
    }
}
pub fn interleave_partitions<T>(data: &mut [T], p: usize) {
    let n = data.len();
    let s = n / p;
    let mut o = 0;
    for _ in 0..p {
        interleave(&mut data[o..o + s]);
        o += s;
    }
}
pub fn riffle_partitions<T>(data: &mut [T], p: usize) {
    let n = data.len();
    let s = n / p;
    let mut o = 0;
    for _ in 0..p {
        riffle(&mut data[o..o + s]);
        o += s;
    }
}
