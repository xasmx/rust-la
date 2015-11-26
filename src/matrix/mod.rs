mod mmatrix;
mod simd;

use std::cmp;
#[cfg(test)]
use std::f32;
use std::fmt::{Formatter, Result};
use std::fmt::Debug;
use std::ops::{Add, BitOr, Index, Mul, Neg, Sub, Range, RangeFrom, RangeFull, RangeTo};
use std::option::Option;
use std::vec::Vec;
use num;
use num::traits::{Float, Num, Signed};
use num::Zero;
use rand;
use rand::Rand;

use ApproxEq;
use decomp::lu;
use decomp::qr;
use internalutil::{alloc_dirty_vec};

//----------------------

#[derive(PartialEq, Clone)]
pub struct Matrix<T> {
  no_rows : usize,
  data : Vec<T>
}

pub trait MatrixRange<IterT : MatrixRangeIterator> {
  fn size(&self, matrix_size : usize) -> usize;
  fn iter(&self) -> IterT;
}

pub trait MatrixRangeIterator {
  fn next(&mut self) -> usize;
}

//----------------------

impl MatrixRangeIterator for usize {
  fn next(&mut self) -> usize { *self }
}

impl MatrixRange<usize> for usize {
  fn size(&self, _matrix_size : usize) -> usize {
    1
  }

  fn iter(&self) -> usize {
    *self
  }
}

//----------------------

pub struct SliceRangeIterator<'a> {
  indexes : &'a [usize],
  slice_index : usize,
  increment : usize
}

impl <'a> MatrixRangeIterator for SliceRangeIterator<'a> {
  fn next(&mut self) -> usize {
    let next_val = self.indexes[self.slice_index];
    self.slice_index += self.increment;
    next_val
  }
}

impl <'a> MatrixRange<SliceRangeIterator<'a>> for &'a [usize] {
  fn size(&self, _matrix_size : usize) -> usize {
    self.len()
  }

  fn iter(&self) -> SliceRangeIterator<'a> {
    SliceRangeIterator {
      indexes : self,
      slice_index : 0,
      increment : 1
    }
  }
}

//----------------------

pub struct RangeIterator {
  index : usize
}

impl MatrixRangeIterator for RangeIterator {
  fn next(&mut self) -> usize {
    let next_val = self.index;
    self.index += 1;
    next_val
  }
}

impl MatrixRange<RangeIterator> for RangeFull {
  fn size(&self, matrix_size : usize) -> usize {
    matrix_size
  }

  fn iter(&self) -> RangeIterator {
    RangeIterator {
      index : 0
    }
  }
}

impl MatrixRange<RangeIterator> for Range<usize> {
  fn size(&self, _matrix_size : usize) -> usize {
    (self.end - self.start) as usize
  }

  fn iter(&self) -> RangeIterator {
    RangeIterator {
      index : self.start
    }
  }
}

impl MatrixRange<RangeIterator> for RangeFrom<usize> {
  fn size(&self, matrix_size : usize) -> usize {
    matrix_size - self.start as usize
  }

  fn iter(&self) -> RangeIterator {
    RangeIterator {
      index : self.start
    }
  }
}

impl MatrixRange<RangeIterator> for RangeTo<usize> {
  fn size(&self, _matrix_size : usize) -> usize {
    self.end as usize
  }

  fn iter(&self) -> RangeIterator {
    RangeIterator {
      index : 0
    }
  }
}

//----------------------

impl<T : Copy> Matrix<T> {
  pub fn new(no_rows : usize, no_cols : usize, data : Vec<T>) -> Matrix<T> {
    assert!(no_rows * no_cols == data.len());
    assert!(no_rows > 0 && no_cols > 0);
    Matrix { no_rows : no_rows, data : data }
  }

  pub fn dirty(no_rows : usize, no_cols : usize) -> Matrix<T> {
    assert!(no_rows > 0 && no_cols > 0);
    let elems = no_rows * no_cols;
    Matrix { no_rows : no_rows, data : alloc_dirty_vec(elems) }
  }

  pub fn vector(data : Vec<T>) -> Matrix<T> {
    assert!(data.len() > 0);
    Matrix { no_rows : data.len(), data : data }
  }

  pub fn row_vector(data : Vec<T>) -> Matrix<T> {
    assert!(data.len() > 0);
    Matrix { no_rows : 1, data : data }
  }

  #[inline]
  pub fn rows(&self) -> usize { self.no_rows }

  #[inline]
  pub fn cols(&self) -> usize { self.data.len() / self.no_rows }

  #[inline]
  pub fn get_data<'a>(&'a self) -> &'a Vec<T> { &self.data }

  pub fn get_ref<'lt>(&'lt self, row : usize, col : usize) -> &'lt T {
    assert!(row < self.no_rows && col < self.cols());
    &self.data[row * self.cols() + col]
  }

  pub fn map<S : Copy>(&self, f : &Fn(&T) -> S) -> Matrix<S> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = f(&self.data[i]);
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn reduce<S : Copy>(&self, init: &Vec<S>, f: &Fn(&S, &T) -> S) -> Matrix<S> {
    assert!(init.len() == self.cols());

    let mut data = init.clone();
    let mut data_idx = 0;
    for i in 0..self.data.len() {
      data[data_idx] = f(&data[data_idx], &self.data[i]);
      data_idx += 1;
      data_idx %= data.len();
    }

    Matrix {
      no_rows : 1,
      data : data
    }
  }

  #[inline]
  pub fn is_square(&self) -> bool {
    self.no_rows == self.cols()
  }

  #[inline]
  pub fn is_not_square(&self) -> bool {
    !self.is_square()
  }
}

impl<T : Num + Copy> Matrix<T> {
  pub fn id(m : usize, n : usize) -> Matrix<T> {
    let elems = m * n;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = num::zero();
    }
    for i in 0..cmp::min(m, n) {
      d[i * n + i] = num::one();
    }
    Matrix { no_rows : m, data : d }
  }

  pub fn zero(no_rows : usize, no_cols : usize) -> Matrix<T> {
    let elems = no_rows * no_cols;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = num::zero();
    }
    Matrix {
      no_rows : no_rows,
      data : d
    }
  }

  pub fn diag(data : Vec<T>) -> Matrix<T> {
    let size = data.len();
    let elems = size * size;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = num::zero();
    }
    for i in 0..size {
      d[i * size + i] = data[i];
    }
    Matrix::new(size, size, d)
  }

  pub fn block_diag(m : usize, n : usize, data : Vec<T>) -> Matrix<T> {
    let min_dim = cmp::min(m, n);
    assert!(data.len() == min_dim);

    let elems = m * n;
    let mut d : Vec<T> = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = num::zero();
    }

    for i in 0..min_dim {
      d[i * n + i] = data[i];
    }
    Matrix::new(m, n, d)
  }

  pub fn zero_vector(no_rows : usize) -> Matrix<T> {
    let mut d : Vec<T> = alloc_dirty_vec(no_rows);
    for i in 0..no_rows {
      d[i] = num::zero();
    }
    Matrix { no_rows : no_rows, data : d }
  }

  pub fn one_vector(no_rows : usize) -> Matrix<T> {
    let mut d : Vec<T> = alloc_dirty_vec(no_rows);
    for i in 0..no_rows {
      d[i] = num::one();
    }
    Matrix { no_rows : no_rows, data : d }
  }
}

impl<T : Num + Neg<Output = T> + Copy> Matrix<T> {
  pub fn scale(&self, factor : T) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = factor * self.data[i];
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn elem_mul(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = self.data[i] * m.data[i];
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn elem_div(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = self.data[i] / m.data[i];
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn dot(&self, m : &Matrix<T>) -> T {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols() && self.cols() == 1);

    let mut sum = num::zero::<T>();
    for i in 0..self.data.len() {
      sum = sum + self.data[i] * m.data[i];
    }
    sum
  }
}


impl<T : Copy> Matrix<T> {
  pub fn get(&self, row : usize, col : usize) -> T {
    assert!(row < self.no_rows && col < self.cols());
    self.data[row * self.cols() + col]
  }

  pub fn cr(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    let mut src_idx1 = 0;
    let mut src_idx2 = 0;
    let mut dest_idx = 0;
    for _ in 0..self.no_rows {
      for _ in 0..self.cols() {
        d[dest_idx] = self.data[src_idx1];
        src_idx1 += 1;
        dest_idx += 1;
      }
      for _ in 0..m.cols() {
        d[dest_idx] = m.data[src_idx2];
        src_idx2 += 1;
        dest_idx += 1;
      }
    }
    Matrix {
      no_rows : self.no_rows,
      data : d
    }
  }

  pub fn cb(&self, m : &Matrix<T>) -> Matrix<T> {
    assert!(self.cols() == m.cols());
    let elems = self.data.len() + m.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..self.data.len() {
      d[i] = self.data[i];
    }
    let offset = self.data.len();
    for i in 0..m.data.len() {
      d[offset + i] = m.data[i];
    }
    Matrix {
      no_rows : self.no_rows + m.no_rows,
      data : d
    }
  }

  pub fn t(&self) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    let mut src_idx = 0;
    for i in 0..elems {
      d[i] = self.data[src_idx];
      src_idx += self.cols();
      if src_idx >= elems {
        src_idx -= elems;
        src_idx += 1;
      }
    }
    Matrix {
      no_rows: self.cols(),
      data : d
    }
  }

  pub fn minor(&self, row : usize, col : usize) -> Matrix<T> {
    assert!(row < self.no_rows && col < self.cols() && self.no_rows > 1 && self.cols() > 1);
    let elems = (self.cols() - 1) * (self.no_rows - 1);
    let mut d = alloc_dirty_vec(elems);
    let mut source_row_idx = 0;
    let mut dest_idx = 0;
    for current_row in 0..self.no_rows {
      if current_row != row {
        for current_col in 0..self.cols() {
          if current_col != col {
            d[dest_idx] = self.data[source_row_idx + current_col];
            dest_idx += 1;
          }
        }
      }
      source_row_idx = source_row_idx + self.cols();
    }
    Matrix {
      no_rows : self.no_rows - 1,
      data : d
    }
  }

  pub fn sub_matrix<RRI, RCI, RR, RC>(&self, rows : RR, cols : RC) -> Matrix<T>
      where RRI : MatrixRangeIterator,
            RCI : MatrixRangeIterator,
            RR : MatrixRange<RRI>,
            RC : MatrixRange<RCI> {
    let no_rows = rows.size(self.rows());
    let no_cols = cols.size(self.cols());
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    let mut dest_idx = 0;
    let mut row_iter = rows.iter();
    for _ in 0..no_rows {
      let row_idx = row_iter.next();
      let src_row_start_idx = row_idx * self.cols();
      let mut col_iter = cols.iter();
      for _ in 0..no_cols {
        let col_idx = col_iter.next();
        d[dest_idx] = self.data[src_row_start_idx + col_idx];
        dest_idx += 1;
      }
    }
    Matrix {
      no_rows : no_rows,
      data : d
    }
  }

  #[inline]
  pub fn get_columns<RCI : MatrixRangeIterator, RC : MatrixRange<RCI>>(&self, columns : RC) -> Matrix<T> {
    self.sub_matrix(.., columns)
  }

  #[inline]
  pub fn get_rows<RCI : MatrixRangeIterator, RC : MatrixRange<RCI>>(&self, row : RC) -> Matrix<T> {
    self.sub_matrix(row, ..)
  }

  #[inline]
  pub fn permute(&self, rows : &[usize], columns : &[usize]) -> Matrix<T> {
    self.sub_matrix(rows, columns)
  }

  #[inline]
  pub fn permute_rows(&self, rows : &[usize]) -> Matrix<T> {
    self.sub_matrix(rows, ..)
  }

  #[inline]
  pub fn permute_columns(&self, columns : &[usize]) -> Matrix<T> {
    self.sub_matrix(.., columns)
  }

  pub fn filter_rows(&self, f : &Fn(&Matrix<T>, usize) -> bool) -> Matrix<T> {
    let mut rows = Vec::with_capacity(self.rows());
    for row in 0..self.rows() {
      if f(self, row) {
        rows.push(row);
      }
    }
    self.permute_rows(&rows)
  }

  pub fn filter_columns(&self, f : &Fn(&Matrix<T>, usize) -> bool) -> Matrix<T> {
    let mut cols = Vec::with_capacity(self.cols());
    for col in 0..self.cols() {
      if f(self, col) {
        cols.push(col);
      }
    }
    self.permute_columns(&cols)
  }

  pub fn select_rows(&self, selector : &[bool]) -> Matrix<T> {
    assert!(self.no_rows == selector.len());
    let mut rows = Vec::with_capacity(self.no_rows);
    for i in 0..selector.len() {
      if selector[i] {
        rows.push(i);
      }
    }
    self.permute_rows(&rows)
  }

  pub fn select_columns(&self, selector : &[bool]) -> Matrix<T> {
    assert!(self.cols() == selector.len());
    let mut cols = Vec::with_capacity(self.cols());
    for i in 0..selector.len() {
      if selector[i] {
        cols.push(i);
      }
    }
    self.permute_columns(&cols)
  }
}

impl<T : Debug + Copy> Matrix<T> {
  pub fn print(&self) {
    print!("{:?}", self);
  }
}

impl<T : Debug + Copy> Debug for Matrix<T> {
  // fmt implementation borrowed (with changes) from matrixrs <https://github.com/doomsplayer/matrixrs>.
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    let max_width =
      self.data.iter().fold(0, |maxlen, elem| {
        let l = format!("{:?}", elem).len();
        if maxlen > l { maxlen } else { l }
      });

    try!(write!(fmt, "\n"));
    for row in 0..self.rows() {
      try!(write!(fmt, "|"));
      for col in 0..self.cols() {
        let v = self.get(row, col);
        let slen = format!("{:?}", v).len();
        let mut padding = " ".to_owned();
        for _ in 0..(max_width-slen) {
          padding.push_str(" ");
        }
        try!(write!(fmt, "{}{:?}", padding, v));
      }
      try!(write!(fmt, " |\n"));
    }
    Ok(())
  }
}

impl<T : Rand + Copy> Matrix<T> {
  pub fn random(no_rows : usize, no_cols : usize) -> Matrix<T> {
    let elems = no_rows * no_cols;
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = rand::random::<T>();
    }
    Matrix { no_rows : no_rows, data : d }
  }
}

impl<'a, T : Neg<Output = T> + Copy> Neg for &'a Matrix<T> {
  type Output = Matrix<T>;

  fn neg(self) -> Matrix<T> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = - self.data[i]
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<T : Neg<Output = T> + Copy> Neg for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn neg(self) -> Matrix<T> { (&self).neg() }
}

impl <'a, 'b, T : Add<T, Output = T> + Copy> Add<&'a Matrix<T>> for &'b Matrix<T> {
  type Output = Matrix<T>;

  fn add(self, m: &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = self.data[i] + m.data[i];
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl <'a, T : Add<T, Output = T> + Copy> Add<Matrix<T>> for &'a Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn add(self, m: Matrix<T>) -> Matrix<T> { self + &m }
}

impl <'a, T : Add<T, Output = T> + Copy> Add<&'a Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn add(self, m: &Matrix<T>) -> Matrix<T> { (&self) + m }
}

impl <T : Add<T, Output = T> + Copy> Add<Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn add(self, m: Matrix<T>) -> Matrix<T> { (&self) + &m }
}

impl <'a, 'b, T : Sub<T, Output = T> + Copy> Sub<&'a Matrix<T>> for &'b Matrix<T> {
  type Output = Matrix<T>;

  fn sub(self, m: &Matrix<T>) -> Matrix<T> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);
    for i in 0..elems {
      d[i] = self.data[i] - m.data[i];
    }
    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl <'a, T : Sub<T, Output = T> + Copy> Sub<Matrix<T>> for &'a Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn sub(self, m: Matrix<T>) -> Matrix<T> { self - &m }
}

impl <'a, T : Sub<T, Output = T> + Copy> Sub<&'a Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn sub(self, m: &Matrix<T>) -> Matrix<T> { (&self) - m }
}

impl <T : Sub<T, Output = T> + Copy> Sub<Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn sub(self, m: Matrix<T>) -> Matrix<T> { (&self) - &m }
}


impl<'a, 'b, T : Add<T, Output = T> + Mul<T, Output = T> + Zero + Copy> Mul<&'a Matrix<T>> for &'b Matrix<T> {
  type Output = Matrix<T>;

  fn mul(self, m: &'a Matrix<T>) -> Matrix<T> {
    assert!(self.cols() == m.no_rows);

    let elems = self.no_rows * m.cols();
    let mut d = alloc_dirty_vec(elems);
    for row in 0..self.no_rows {
      for col in 0..m.cols() {
        let mut res : T = num::zero();
        for idx in 0..self.cols() {
          res = res + self.get(row, idx) * m.get(idx, col);
        }
        d[row * m.cols() + col] = res;
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}

impl<'a, T : Add<T, Output = T> + Mul<T, Output = T> + Zero + Copy> Mul<Matrix<T>> for &'a Matrix<T> {
  type Output = Matrix<T>;

  fn mul(self, m: Matrix<T>) -> Matrix<T> { self * &m }
}

impl<T : Add<T, Output = T> + Mul<T, Output = T> + Zero + Copy> Mul<Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  fn mul(self, m: Matrix<T>) -> Matrix<T> { (&self) * &m }
}

impl<'a, T : Add<T, Output = T> + Mul<T, Output = T> + Zero + Copy> Mul<&'a Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  fn mul(self, m: &'a Matrix<T>) -> Matrix<T> { (&self) * m }
}

impl<T : Copy> Index<(usize, usize)> for Matrix<T> {
  type Output = T;

  #[inline]
  fn index<'a>(&'a self, (y, x): (usize, usize)) -> &'a T { self.get_ref(y, x) }
}

impl<'a, T : Copy> BitOr<&'a Matrix<T>> for Matrix<T> {
  type Output = Matrix<T>;

  #[inline]
  fn bitor(self, rhs: &Matrix<T>) -> Matrix<T> { self.cr(rhs) }
}

impl<T : Float + ApproxEq<T> + Signed + Copy> Matrix<T> {
  pub fn trace(&self) -> T {
    let mut sum : T = num::zero();
    let mut idx = 0;
    for _ in 0..cmp::min(self.no_rows, self.cols()) {
      sum = sum + self.data[idx];
      idx += self.cols() + 1;
    }
    sum
  }

  pub fn det(&self) -> T {
    assert!(self.cols() == self.no_rows);
    lu::LUDecomposition::new(self).det()
  }

  pub fn solve(&self, b : &Matrix<T>) -> Option<Matrix<T>> {
    lu::LUDecomposition::new(self).solve(b)
  }

  pub fn inverse(&self) -> Option<Matrix<T>> {
    assert!(self.no_rows == self.cols());
    lu::LUDecomposition::new(self).solve(&Matrix::id(self.no_rows, self.no_rows))
  }

  #[inline]
  pub fn is_singular(&self) -> bool {
    !self.is_non_singular()
  }

  pub fn is_non_singular(&self) -> bool {
    assert!(self.no_rows == self.cols());
    lu::LUDecomposition::new(self).is_non_singular()
  }

  pub fn pinverse(&self) -> Matrix<T> {
    // A+ = (A' A)^-1 A'
    //    = ((QR)' QR)^-1 A'
    //    = (R'Q'QR)^-1 A'
    //    = (R'R)^-1 A'
    let qr = qr::QRDecomposition::new(self);
    let r = qr.get_r();
    (r.t() * &r).inverse().unwrap() * &self.t()
  }

  pub fn vector_euclidean_norm(&self) -> T {
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in 0..self.data.len() {
      s = s + self.data[i] * self.data[i];
    }

    s.sqrt()
  }

  #[inline]
  pub fn length(&self) -> T {
    self.vector_euclidean_norm()
  }

  pub fn vector_1_norm(&self) -> T {
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in 0..self.data.len() {
      s = s + num::abs(self.data[i]);
    }

    s
  }

  #[inline]
  pub fn vector_2_norm(&self) -> T {
    self.vector_euclidean_norm()
  }

  pub fn vector_p_norm(&self, p : T) -> T {
    assert!(self.cols() == 1);

    let mut s : T = num::zero();
    for i in 0..self.data.len() {
      s = s + num::abs(self.data[i].powf(p));
    }

    s.powf(num::one::<T>() / p)
  }

  pub fn frobenius_norm(&self) -> T {
    let mut s : T = num::zero();
    for i in 0..self.data.len() {
      s = s + self.data[i] * self.data[i];
    }

    s.sqrt()
  }

  pub fn vector_inf_norm(&self) -> T {
    assert!(self.cols() == 1);

    let mut current_max : T = num::abs(self.data[0]);
    for i in 1..self.data.len() {
      let v = num::abs(self.data[i]);
      if v > current_max {
        current_max = v;
      }
    }

    current_max
  }

  pub fn is_symmetric(&self) -> bool {
    if self.no_rows != self.cols() { return false; }
    for row in 1..self.no_rows {
      for col in 0..row {
        if !self.get(row, col).approx_eq(self.get_ref(col, row)) { return false; }
      }
    }

    true
  }

  #[inline]
  pub fn is_non_symmetric(&self) -> bool {
    !self.is_symmetric()
  }

  pub fn approx_eq(&self, m : &Matrix<T>) -> bool {
    if self.rows() != m.rows() || self.cols() != m.cols() { return false };
    for i in 0..self.data.len() {
      if !self.data[i].approx_eq(&m.data[i]) { return false }
    }
    true
  }
}

#[test]
fn test_new() {
  let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
  m.print();
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 2, 3, 4]);
}

#[test]
#[should_panic]
fn test_new_invalid_data() {
  Matrix::new(1, 2, vec![1, 2, 3]);
}

#[test]
#[should_panic]
fn test_new_invalid_row_count() {
  Matrix::<usize>::new(0, 2, vec![]);
}

#[test]
#[should_panic]
fn test_new_invalid_col_count() {
  Matrix::<usize>::new(2, 0, vec![]);
}

#[test]
fn test_dirty() {
  let m : Matrix<usize> = Matrix::dirty(3, 2);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
}

#[test]
fn test_id_square() {
  let m = Matrix::<usize>::id(2, 2);
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 1]);
}

#[test]
fn test_id_m_over_n() {
  let m = Matrix::<usize>::id(3, 2);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 1, 0, 0]);
}

#[test]
fn test_id_n_over_m() {
  let m = Matrix::<usize>::id(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == vec![1, 0, 0, 0, 1, 0]);
}

#[test]
fn test_zero() {
  let m = Matrix::<usize>::zero(2, 3);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_diag() {
  let m = Matrix::<usize>::diag(vec![1, 2]);
  assert!(m.rows() == 2);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 2]);
}

#[test]
fn test_block_diag() {
  let m = Matrix::<usize>::block_diag(2, 3, vec![1, 2]);
  assert!(m.rows() == 2);
  assert!(m.cols() == 3);
  assert!(m.data == vec![1, 0, 0, 0, 2, 0]);

  let m = Matrix::<usize>::block_diag(3, 2, vec![1, 2]);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 0, 0, 2, 0, 0]);
}

#[test]
fn test_vector() {
  let v = Matrix::vector(vec![1, 2, 3]);
  assert!(v.rows() == 3);
  assert!(v.cols() == 1);
  assert!(v.data == vec![1, 2, 3]);
}

#[test]
fn test_zero_vector() {
  let v = Matrix::<usize>::zero_vector(2);
  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == vec![0, 0]);
}

#[test]
fn test_one_vector() {
  let v = Matrix::<usize>::one_vector(2);

  assert!(v.rows() == 2);
  assert!(v.cols() == 1);
  assert!(v.data == vec![1, 1]);
}

#[test]
fn test_row_vector() {
  let v = Matrix::row_vector(vec![1, 2, 3]);
  assert!(v.rows() == 1);
  assert!(v.cols() == 3);
  assert!(v.data == vec![1, 2, 3]);
}

#[test]
fn test_get() {
  let m = m!(1, 2; 3, 4);
  assert!(m.get(1, 0) == 3);
  assert!(m.get(0, 1) == 2);

  assert!(*m.get_ref(1, 1) == 4);
}

#[test]
#[should_panic]
fn test_get_out_of_bounds_x() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get(2, 0);
}

#[test]
#[should_panic]
fn test_get_out_of_bounds_y() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get(0, 2);
}

#[test]
#[should_panic]
fn test_get_ref_out_of_bounds_x() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get_ref(2, 0);
}

#[test]
#[should_panic]
fn test_get_ref_out_of_bounds_y() {
  let m = m!(1, 2; 3, 4);
  let _ = m.get_ref(0, 2);
}

#[test]
fn test_map() {
  let m = m!(1, 2; 3, 4);
  assert!(m.map(&|x : &usize| -> usize { *x + 1 }).data == vec![2, 3, 4, 5]);
}

#[test]
fn test_cr() {
  let v = m!(1; 2; 3);
  let m = v.cr(&v);
  assert!(m.rows() == 3);
  assert!(m.cols() == 2);
  assert!(m.data == vec![1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_cb() {
  let m = m!(1, 2; 3, 4);
  let m2 = m.cb(&m);
  assert!(m2.rows() == 4);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![1, 2, 3, 4, 1, 2, 3, 4]);
}

#[test]
fn test_t() {
  let m = m!(1, 2; 3, 4);
  assert!(m.t().data == vec![1, 3, 2, 4]);

  let m = m!(1, 2, 3; 4, 5, 6);
  let r = m.t();
  assert!(r.rows() == 3);
  assert!(r.cols() == 2);
  assert!(r.data == vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_sub() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.sub_matrix(1..3, 1..3).data == vec![5, 6, 8, 9]);
}

#[test]
#[should_panic]
fn test_sub_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.sub_matrix(1..3, 1..4);
}

#[test]
fn test_minor() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.minor(1, 1).data == vec![1, 3, 7, 9]);
}

#[test]
#[should_panic]
fn test_minor_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.minor(1, 4);
}

#[test]
fn test_get_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.get_columns(1).data == vec![2, 5, 8]);
}

#[test]
#[should_panic]
fn test_get_columns_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.get_columns(3);
}

#[test]
fn test_get_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.get_rows(1).data == vec![4, 5, 6]);
}

#[test]
#[should_panic]
fn test_get_rows_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.get_rows(3);
}

#[test]
fn test_permute_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.permute_rows(&[1, 0, 2]).data == vec![4, 5, 6, 1, 2, 3, 7, 8, 9]);
  assert!(m.permute_rows(&[2, 1]).data == vec![7, 8, 9, 4, 5, 6]);
}

#[test]
#[should_panic]
fn test_permute_rows_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.permute_rows(&[1, 0, 5]);
}

#[test]
fn test_permute_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  assert!(m.permute_columns(&[1, 0, 2]).data == vec![2, 1, 3, 5, 4, 6, 8, 7, 9]);
  assert!(m.permute_columns(&[1, 2]).data == vec![2, 3, 5, 6, 8, 9]);
}

#[test]
#[should_panic]
fn test_permute_columns_out_of_bounds() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let _ = m.permute_columns(&[1, 0, 5]);
}

#[test]
fn test_filter_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.filter_rows(&|_, row| { ((row % 2) == 0) });
  assert!(m2.rows() == 2);
  assert!(m2.cols() == 3);
  assert!(m2.data == vec![1, 2, 3, 7, 8, 9]); 
}

#[test]
fn test_filter_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.filter_columns(&|_, col| { (col >= 1) });
  m2.print();
  assert!(m2.rows() == 3);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![2, 3, 5, 6, 8, 9]); 
}

#[test]
fn test_select_rows() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.select_rows(&[false, true, true]);
  assert!(m2.rows() == 2);
  assert!(m2.cols() == 3);
  assert!(m2.data == vec![4, 5, 6, 7, 8, 9]); 
}

#[test]
fn test_select_columns() {
  let m = m!(1, 2, 3; 4, 5, 6; 7, 8, 9);
  let m2 = m.select_columns(&[true, false, true]);
  assert!(m2.rows() == 3);
  assert!(m2.cols() == 2);
  assert!(m2.data == vec![1, 3, 4, 6, 7, 9]); 
}

#[test]
fn test_algebra() {
  let a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  assert!((&a).neg().data == vec![-1, -2, -3, -4]);
  assert!((&a).scale(2).data == vec![2, 4, 6, 8]);
  assert!((&a).add(&b).data == vec![4, 6, 8, 10]);
  assert!((&b).sub(&a).data == vec![2, 2, 2, 2]);
  assert!((&a).elem_mul(&b).data == vec![3, 8, 15, 24]);
  assert!((&b).elem_div(&a).data == vec![3, 2, 1, 1]);
}

#[test]
fn test_dot() {
  let a = m!(1; 2; 3; 4);
  let b = m!(3; 4; 5; 6);
  assert!((&a).dot(&b) == 50);
}

#[test]
fn test_mul() {
  let a = m!(1, 2; 3, 4);
  let b = m!(3, 4; 5, 6);
  assert!((&a).mul(&b).data == vec![13, 16, 29, 36]);
}

#[test]
#[should_panic]
fn test_mul_incompatible() {
  let a = m!(1, 2; 3, 4);
  let b = m!(1, 2; 3, 4; 5, 6);
  let _ = (&a).mul(&b);
}

#[test]
fn test_trace() {
  let a = m!(1.0, 2.0; 3.0, 4.0);
  assert!(a.trace() == 5.0);

  let a = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0);
  assert!(a.trace() == 5.0);

  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0);
  assert!(a.trace() == 6.0);
}

#[test]
fn test_det() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0; 0.0, 5.0, -7.0);
  assert!((a.det() - -96.0) <= f32::EPSILON);
}

#[test]
#[should_panic]
fn test_det_not_square() {
  let _ = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0).det();
}

#[test]
fn test_solve() {
  let a = m!(1.0, 1.0, 1.0; 1.0, -1.0, 4.0; 2.0, 3.0, -5.0);
  let b = m!(3.0; 4.0; 0.0);
  assert!(a.solve(&b).unwrap().eq(&m!(1.0; 1.0; 1.0)));
}

// TODO: Add more tests for solve

#[test]
fn test_inverse() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0; 0.0, 5.0, -7.0);
  let data : Vec<f64> = vec![16.0, -1.0, 23.0, 0.0, 42.0, -6.0, 0.0, 30.0, -18.0].iter_mut().map(|x : &mut f64| -> f64 { *x / 96.0 }).collect();
  let a_inv = Matrix::new(3, 3, data);
  assert!(a.inverse().unwrap().approx_eq(&a_inv));
}

#[test]
#[should_panic]
fn test_inverse_not_square() {
  let a = m!(6.0, -7.0, 10.0; 0.0, 3.0, -1.0);
  let _ = a.inverse();
}

#[test]
fn test_inverse_singular() {
  let a = m!(2.0, 6.0; 1.0, 3.0);
  assert!(a.inverse() == None);
}

#[test]
fn test_pinverse() {
  let a = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0);
  assert!((a.pinverse() * a).approx_eq(&Matrix::<f64>::id(2, 2)));
}

#[test]
fn test_is_singular() {
  let m = m!(2.0, 6.0; 1.0, 3.0);
  assert!(m.is_singular());
}

#[test]
#[should_panic]
fn test_is_singular_non_square() {
  let m = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0);
  assert!(m.is_singular());
}

#[test]
fn test_is_non_singular() {
  let m = m!(2.0, 6.0; 6.0, 3.0);
  assert!(m.is_non_singular());
}

#[test]
fn test_is_square() {
  let m = m!(1, 2; 3, 4);
  assert!(m.is_square());
  assert!(!m.is_not_square());

  let m = m!(1, 2, 3; 4, 5, 6);
  assert!(!m.is_square());
  assert!(m.is_not_square());

  let v = m!(1; 2; 3);
  assert!(!v.is_square());
  assert!(v.is_not_square());
}

#[test]
fn test_is_symmetric() {
  let m = m!(1.0, 2.0, 3.0; 2.0, 4.0, 5.0; 3.0, 5.0, 6.0);
  assert!(m.is_symmetric());

  let m = m!(1.0, 2.0; 3.0, 4.0);
  assert!(!m.is_symmetric());

  let m = m!(1.0, 2.0, 3.0; 2.0, 4.0, 5.0);
  assert!(!m.is_symmetric());
}

#[test]
fn test_vector_euclidean_norm() {
  assert!(m!(1.0; 2.0; 2.0).vector_euclidean_norm() == 3.0);
  assert!(m!(-2.0; 2.0; 2.0; 2.0).vector_euclidean_norm() == 4.0);
}

#[test]
#[should_panic]
fn test_vector_euclidean_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_euclidean_norm();
}

#[test]
fn test_vector_1_norm() {
  assert!(m!(-3.0; 2.0; 2.5).vector_1_norm() == 7.5);
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_1_norm() == 19.0);
  assert!(m!(1.0).vector_1_norm() == 1.0);
}

#[test]
#[should_panic]
fn test_vector_1_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_1_norm();
}

#[test]
fn test_vector_p_norm() {
  assert!(m!(-3.0; 2.0; 2.0).vector_p_norm(3.0) == 43.0f64.powf(1.0 / 3.0));
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_p_norm(5.0) == 40819.0f64.powf(1.0 / 5.0));
  assert!(m!(1.0).vector_p_norm(2.0) == 1.0);
}

#[test]
#[should_panic]
fn test_vector_p_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_p_norm(1.0);
}

#[test]
fn test_vector_inf_norm() {
  assert!(m!(-3.0; 2.0; 2.5).vector_inf_norm() == 3.0);
  assert!(m!(6.0; 8.0; -2.0; 3.0).vector_inf_norm() == 8.0);
  assert!(m!(1.0).vector_inf_norm() == 1.0);
}

#[test]
#[should_panic]
fn test_vector_inf_norm_not_vector() {
  let _ = m!(1.0, 2.0; 3.0, 4.0).vector_inf_norm();
}

#[test]
fn test_frobenius_norm() {
  assert!(m!(1.0, 2.0; 3.0, 4.0).frobenius_norm() == 30.0f64.sqrt());
  assert!(m!(1.0; 2.0; 2.0).frobenius_norm() == 3.0);
}

