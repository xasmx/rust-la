use std::io;
use std::num;
use std::num::{Zero, One};
use std::path;
use std::vec;

#[inline]
pub fn alloc_dirty_vec<T>(n : uint) -> ~[T] {
  let mut v : ~[T] = vec::with_capacity(n);
  unsafe {
    vec::raw::set_len(&mut v, n);
  }
  v
}

// Ported from JAMA.
// sqrt(a^2 + b^2) without under/overflow.
pub fn hypot<T : Zero + One + Signed + Algebraic + Orderable + Clone>(a : T, b : T) -> T {
  if num::abs(a.clone()) > num::abs(b.clone()) {
    let r = b / a;
    return num::abs(a.clone()) * num::sqrt(num::one::<T>() + r * r);
  } else if b != Zero::zero() {
    let r = a / b;
    return num::abs(b.clone()) * num::sqrt(num::one::<T>() + r * r);
  } else {
    return Zero::zero();
  }
}

pub fn read_csv<T>(file_name : &str, parser : &fn(&str) -> T) -> super::matrix::Matrix<T> {
  let mut data = vec::with_capacity(16384);
  let mut row_count = 0;
  let mut col_count = None;
  match(io::file_reader(&path::Path(file_name))) {
    Ok(reader) => {
      do reader.each_line |line| {
        let element_count = data.len();
        for item in line.split_iter(',') {
          data.push(parser(item))
        }
        let line_col_count = data.len() - element_count;

        if(col_count == None) {
          col_count = Some(line_col_count);
        } else {
          assert!(col_count.unwrap() == line_col_count);
        }

        row_count += 1;
        true
      }
    }
    _ => { fail!(~"Failed to load file.") }
  };

  assert!(col_count != None);

  super::matrix::matrix(row_count, col_count.unwrap(), data)
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
