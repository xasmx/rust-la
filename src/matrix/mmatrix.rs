use std::ops::{Neg};
use std::vec::Vec;
use num;
use num::traits::{Num};

use matrix::Matrix;


impl<T : Copy> Matrix<T> {
  #[inline]
  pub fn get_mut_data<'a>(&'a mut self) -> &'a mut Vec<T> { &mut self.data }

  pub fn get_mref<'lt>(&'lt mut self, row : usize, col : usize) -> &'lt mut T {
    assert!(row < self.no_rows && col < self.cols());
    let no_cols = self.cols();
    &mut self.data[row * no_cols + col]
  }

  pub fn mmap<'a>(&'a mut self, f : &Fn(&T) -> T) -> &'a mut Matrix<T> {
    for i in 0..self.data.len() {
      self.data[i] = f(&self.data[i]);
    }
    self
  }
}

impl<T : Num + Neg<Output = T> + Copy> Matrix<T> {
  pub fn mneg<'a>(&'a mut self) -> &'a mut Matrix<T> {
    for i in 0..self.data.len() {
      self.data[i] = - self.data[i];
    }
    self
  }

  pub fn mscale<'a> (&'a mut self, factor : T) -> &'a mut Matrix<T> {
    for i in 0..self.data.len() {
      self.data[i] = factor * self.data[i];
    }
    self
  }

  pub fn madd<'a>(&'a mut self, m : &Matrix<T>) -> &'a mut Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in 0..self.data.len() {
      self.data[i] = self.data[i] + m.data[i];
    }
    self
  }

  pub fn msub<'a>(&'a mut self, m : &Matrix<T>) -> &'a mut Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in 0..self.data.len() {
      self.data[i] = self.data[i] - m.data[i]
    }
    self
  }

  pub fn melem_mul<'a>(&'a mut self, m : &Matrix<T>) -> &'a mut Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in 0..self.data.len() {
      self.data[i] = self.data[i] * m.data[i];
    }
    self
  }

  pub fn melem_div<'a>(&'a mut self, m : &Matrix<T>) -> &'a mut Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    for i in 0..self.data.len() {
      self.data[i] = self.data[i] / m.data[i];
    }
    self
  }

  pub fn mmul<'a>(&self, m : &Matrix<T>, dst : &'a mut Matrix<T>) -> &'a mut Matrix<T> {
    assert!(self.cols() == m.no_rows);
    assert!(dst.rows() == self.no_rows);
    assert!(dst.cols() == m.cols());

    for row in 0..self.no_rows {
      for col in 0..m.cols() {
        let mut res : T = num::zero();
        for idx in 0..self.cols() {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        dst.data[row * m.cols() + col] = res;
      }
    }

    dst
  }
}


impl<T : Copy> Matrix<T> {
  pub fn set<'a>(&'a mut self, row : usize, col : usize, val : T) -> &'a mut Matrix<T> {
    assert!(row < self.no_rows && col < self.cols());
    let no_cols = self.cols();
    self.data[row * no_cols + col] = val;
    self
  }

  pub fn mt<'a>(&'a mut self) -> &'a mut Matrix<T> {
    let mut visited = vec![false; self.data.len()];

    for cycle_idx in 1..(self.data.len() - 1) {
      if visited[cycle_idx] {
        continue;
      }

      let mut idx = cycle_idx;
      let mut prev_value = self.data[idx];
      loop {
        idx = (self.no_rows * idx) % (self.data.len() - 1);
        let current_value = self.data[idx];
        self.data[idx] = prev_value;
        if idx == cycle_idx {
          break;
        }

        prev_value = current_value;
        visited[idx] = true;
      }
    }

    self.no_rows = self.cols();
    self
  }
}

#[test]
fn test_get_set() {
  let mut m = m!(1, 2; 3, 4);
  *m.get_mref(0, 0) = 10;
  assert!(m.get(0, 0) == 10);

  m.set(1, 1, 5);
  assert!(m.get(1, 1) == 5);
}

#[test]
#[should_panic]
fn test_get_mref_out_of_bounds_x() {
  let mut m = m!(1, 2; 3, 4);
  let _ = m.get_mref(2, 0);
}

#[test]
#[should_panic]
fn test_get_mref_out_of_bounds_y() {
  let mut m = m!(1, 2; 3, 4);
  let _ = m.get_mref(0, 2);
}

#[test]
#[should_panic]
fn test_set_out_of_bounds_x() {
  let mut m = m!(1, 2; 3, 4);
  m.set(2, 0, 0);
}

#[test]
#[should_panic]
fn test_set_out_of_bounds_y() {
  let mut m = m!(1, 2; 3, 4);
  m.set(0, 2, 0);
}

#[test]
fn test_map() {
  let mut m = m!(1, 2; 3, 4);
  m.mmap(&|x : &usize| { *x + 2 });
  assert!(m.data == vec![3, 4, 5, 6]);
}

#[test]
fn test_t() {
  let mut m = m!(1, 2; 3, 4);
  m.mt();
  assert!(m.data == vec![1, 3, 2, 4]);

  let mut m = m!(1, 2, 3; 4, 5, 6);
  m.mt();
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_algebra() {
  let mut a = m!(1, 2; 3, 4);
  a.mneg();
  assert!(a.data == vec![-1, -2, -3, -4]);

  let mut a = m!(1, 2; 3, 4);
  a.mscale(2);
  assert!(a.data == vec![2, 4, 6, 8]);

  let mut a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  a.madd(&b);
  assert!(a.data == vec![4, 6, 8, 10]);

  let a = m!(1, 2; 3, 4);
  let mut b = m!(3, 4; 5, 6);
  b.msub(&a);
  assert!(b.data == vec![2, 2, 2, 2]);

  let mut a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  a.melem_mul(&b);
  assert!(a.data == vec![3, 8, 15, 24]);

  let a = m!(1, 2; 3, 4);
  let mut b = m!(3, 4; 5, 6);
  b.melem_div(&a);
  assert!(b.data == vec![3, 2, 1, 1]);

  let mut a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  a.mneg()
   .mscale(2)
   .madd(&b);
  assert!(a.data == vec![1, 0, -1, -2]);
}

#[test]
fn test_mmul() {
  let a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  let mut dst = Matrix::dirty(2, 2);
  a.mmul(&b, &mut dst);
  assert!(dst.data == vec![13, 16, 29, 36]);
}

#[test]
#[should_panic]
fn test_mmul_incompatible_src() {
  let a = m!(1, 2; 3, 4);
  let b = m!(1, 2; 3, 4; 5, 6);
  let mut dst = Matrix::dirty(2, 2);
  a.mmul(&b, &mut dst);
}

#[test]
#[should_panic]
fn test_mmul_incompatible_dst() {
  let a = m!(1, 2; 3, 4);
  let b = m!(1, 2; 3, 4);
  let mut dst = Matrix::dirty(3, 3);
  a.mmul(&b, &mut dst);
}

