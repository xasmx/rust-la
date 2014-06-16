#![macro_escape]

/// Macro for building matrices.
///
/// # Example
///
/// ```
/// #![feature(phase)]
/// #[phase(plugin, link)] extern crate la;
///
/// ...
/// # fn main() {
/// let _m = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0);
/// # }
/// ```

/// Helper macro for m!
#[macro_export]
macro_rules! m_one {
  ( $item:tt ) => ( 1 )
}

/// Helper macro for m!
#[macro_export]
macro_rules! m_rec(
  ( [ $($row:tt),* ] [$($i:expr),*]) => ({
     let _rows = 0 $(+ m_one!($row) )*;
     let _cols = (0 $(+ m_one!($i))*) / _rows;
     Matrix::new(
       _rows,
       _cols,
       vec![$($i),*]
     )
  })
)

/// Macro for building matrices.
#[macro_export]
macro_rules! m {
  ( $( $( $i:expr ),* );* ) => ( m_rec!([$([$($i),*]),*] [$($($i),*),*]) )
}
