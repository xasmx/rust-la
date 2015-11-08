use std::vec;

use num;
use num::{Float, Signed, One, Zero};

#[inline]
pub fn alloc_dirty_vec<T : Copy>(n : usize) -> Vec<T> {
  let mut v : Vec<T> = vec::Vec::with_capacity(n);
  unsafe {
    v.set_len(n);
  }
  v
}

// Ported from JAMA.
// sqrt(a^2 + b^2) without under/overflow.
pub fn hypot<T : Zero + One + Signed + PartialOrd + Float + Clone>(a : T, b : T) -> T {
  if num::abs(a) > num::abs(b) {
    let r = b / a;
    return num::abs(a) * (num::one::<T>() + r * r).sqrt();
  } else if b != Zero::zero() {
    let r = a / b;
    return num::abs(b) * (num::one::<T>() + r * r).sqrt();
  } else {
    return Zero::zero();
  }
}

#[test]
fn hypot_test() {
  assert!(hypot(3.0, 4.0) == 5.0);
  assert!(hypot(4.0, 3.0) == 5.0);
  assert!(hypot(4.0, 0.0) == 4.0);
  assert!(hypot(0.0, 4.0) == 4.0);
  assert!(hypot(-3.0, 4.0) == 5.0);
  assert!(hypot(4.0, -3.0) == 5.0);
  assert!(hypot(-4.0, -3.0) == 5.0);
  assert!(hypot(-4.0, 0.0) == 4.0);
  assert!(hypot(0.0, -4.0) == 4.0);
}
