use simd::f32x4;

use matrix::Matrix;
use internalutil::{alloc_dirty_vec};

impl Matrix<f32> {
  pub fn scale(self, factor : f32) -> Matrix<f32> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    let factor_lanes = f32x4::splat(factor);
    for i in 0..(self.data.len() >> 2) {
      let idx = i << 2;
      let res = factor_lanes * f32x4::load(&self.data as &[f32], idx);
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = self.data.len() & !3;
    if extra_elems != 0 {
      for i in (self.data.len() - extra_elems) .. self.data.len() {
        d[i] = factor * self.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn add(self, m: &Matrix<f32>) -> Matrix<f32> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    for i in 0..(m.data.len() >> 2) {
      let idx = i << 2;
      let a = f32x4::load(&self.data as &[f32], idx);
      let b = f32x4::load(&m.data as &[f32], idx);
      let res = a + b;
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = m.data.len() & !3;
    if extra_elems != 0 {
      for i in (m.data.len() - extra_elems) .. m.data.len() {
        d[i] = self.data[i] + m.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn sub(self, m: &Matrix<f32>) -> Matrix<f32> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    for i in 0..(m.data.len() >> 2) {
      let idx = i << 2;
      let a = f32x4::load(&self.data as &[f32], idx);
      let b = f32x4::load(&m.data as &[f32], idx);
      let res = a - b;
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = m.data.len() & !3;
    if extra_elems != 0 {
      for i in (m.data.len() - extra_elems) .. m.data.len() {
        d[i] = self.data[i] - m.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn elem_mul(self, m: &Matrix<f32>) -> Matrix<f32> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    for i in 0..(m.data.len() >> 2) {
      let idx = i << 2;
      let a = f32x4::load(&self.data as &[f32], idx);
      let b = f32x4::load(&m.data as &[f32], idx);
      let res = a * b;
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = m.data.len() & !3;
    if extra_elems != 0 {
      for i in (m.data.len() - extra_elems) .. m.data.len() {
        d[i] = self.data[i] * m.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn elem_div(self, m: &Matrix<f32>) -> Matrix<f32> {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols());

    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    for i in 0..(m.data.len() >> 2) {
      let idx = i << 2;
      let a = f32x4::load(&self.data as &[f32], idx);
      let b = f32x4::load(&m.data as &[f32], idx);
      let res = a / b;
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = m.data.len() & !3;
    if extra_elems != 0 {
      for i in (m.data.len() - extra_elems) .. m.data.len() {
        d[i] = self.data[i] / m.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn neg(self) -> Matrix<f32> {
    let elems = self.data.len();
    let mut d = alloc_dirty_vec(elems);

    for i in 0..(self.data.len() >> 2) {
      let idx = i << 2;
      let res = - f32x4::load(&self.data as &[f32], idx);
      res.store(&mut d as &mut [f32], idx);
    }

    let extra_elems = self.data.len() & !3;
    if extra_elems != 0 {
      for i in (self.data.len() - extra_elems) .. self.data.len() {
        d[i] = - self.data[i];
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }

  pub fn dot(&self, m : &Matrix<f32>) -> f32 {
    assert!(self.no_rows == m.no_rows);
    assert!(self.cols() == m.cols() && self.cols() == 1);

    let mut sum_lanes = f32x4::splat(0.0);
    for i in 0..(self.data.len() >> 2) {
      let idx = i << 2;
      let a = f32x4::load(&self.data as &[f32], idx);
      let b = f32x4::load(&m.data as &[f32], idx);
      sum_lanes = sum_lanes + (a * b);
    }

    let mut sum = sum_lanes.extract(0) + sum_lanes.extract(1) + sum_lanes.extract(2) + sum_lanes.extract(3);
    let extra_elems = self.data.len() & !3;
    if extra_elems != 0 {
      for i in (self.data.len() - extra_elems) .. self.data.len() {
        sum += self.data[i];
      }
    }

    sum
  }

  pub fn mul(&self, m2: &Matrix<f32>) -> Matrix<f32> {
    assert!(self.cols() == m2.rows());
    let elems = self.rows() * m2.cols();
    let mut d = alloc_dirty_vec(elems);

    let m2_cols = m2.cols();
    for row in 0..self.rows() {
      let m1_row_start_idx = row * self.cols();
      for col_block in 0..(m2_cols / 4) {
        let m2_col_start_idx = col_block * 4;
        let mut res = f32x4::splat(0.0);
        for i in 0..self.cols() {
          let a = f32x4::load(&m2.data as &[f32], i * m2_cols + m2_col_start_idx);
          res = res + f32x4::splat(self.data[m1_row_start_idx + i]) * a;
        }
        res.store(&mut d as &mut [f32], row * m2_cols + m2_col_start_idx);
      }
      for col in (m2_cols & !3)..m2_cols {
        let mut res = 0.0;
        for i in 0..self.cols() {
          res += self.data[m1_row_start_idx + i] * m2.data[i * m2_cols + col];
        }
        d[row * m2_cols + col] = res;
      }
    }

    Matrix {
      no_rows: self.no_rows,
      data : d
    }
  }
}
