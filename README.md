# rust-la

[![Build Status](https://travis-ci.org/xasmx/rust-la.svg?branch=master)](https://travis-ci.org/xasmx/rust-la)

Linear algebra library for the Rust programming language.

## Documentation

See [here](http://xasmx.github.io/rust-la/doc/la/index.html).

## Usage

To use this crate, add la as a dependency to your project's Cargo.toml:

```
[dependencies]
la = "0.2.0"
```

## Example

Here is an example rust program using la to create matrices, perform
basic matrix operations, and perform a Singular Value Decomposition.

```
#[macro_use] extern crate la;

use la::{Matrix, SVD};

fn main() {
    let a = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0);
    let b = m!(7.0, 8.0, 9.0; 10.0, 11.0, 12.0);
    let c = (a * b).t();
    println!("{:?}", c);

    let svd = SVD::new(&c);
    println!("{:?}", svd.get_s());
}
```

## Features

* BLAS
* immutable and mutable implementations
* inverse, solve
* decompositions, including
  * Cholesky
  * LU
  * QR
  * Eigen
  * SVD.

Only dense matrixes.

