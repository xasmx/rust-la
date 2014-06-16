use std::io;
use std::num;
use std::num::{Zero, One};
use std::path;
use std::vec;

#[inline]
pub fn alloc_dirty_vec<T>(n : uint) -> Vec<T> {
  let mut v : Vec<T> = vec::Vec::with_capacity(n);
  unsafe {
    v.set_len(n);
  }
  v
}

// Ported from JAMA.
// sqrt(a^2 + b^2) without under/overflow.
pub fn hypot<T : Zero + One + Signed + PartialOrd + Float + Clone>(a : T, b : T) -> T {
  if num::abs(a.clone()) > num::abs(b.clone()) {
    let r = b / a;
    return num::abs(a.clone()) * (num::one::<T>() + r * r).sqrt();
  } else if b != Zero::zero() {
    let r = a / b;
    return num::abs(b.clone()) * (num::one::<T>() + r * r).sqrt();
  } else {
    return Zero::zero();
  }
}

pub fn read_csv<T>(file_name : &str, parser : |&str| -> T) -> super::matrix::Matrix<T> {
  let mut data = vec::Vec::with_capacity(16384);
  let mut row_count = 0;
  let mut col_count = None;

  let path = path::Path::new(file_name);
  let mut file = io::BufferedReader::new(io::File::open(&path));
  for line in file.lines() {
    let element_count = data.len();
    for item in line.unwrap().as_slice().split_str(",") {
      data.push(parser(item.trim()))
    }
    let line_col_count = data.len() - element_count;

    if col_count == None {
      col_count = Some(line_col_count);
    } else {
      assert!(col_count.unwrap() == line_col_count);
    }

    row_count += 1;
  }

  assert!(col_count != None);

  super::matrix::Matrix::new(row_count, col_count.unwrap(), data)
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
