use std::vec;

#[inline]
pub fn alloc_dirty_vec<T>(n : uint) -> ~[T] {
  let mut v : ~[T] = vec::with_capacity(n);
  unsafe {
    vec::raw::set_len(&mut v, n);
  }
  v
}
