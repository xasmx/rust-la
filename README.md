# rust-la

[![Build Status](https://travis-ci.org/xasmx/rust-la?branch=master)](https://travis-ci.org/xasmx/rust-la)

Linear algebra library for the Rust programming language.

## Documentation

See [here](http://xasmx.github.io/rust-la/doc/la/index.html).

## Usage

To use this crate, add la as a dependency to your project's Cargo.toml:

```
[dependencies]
la = "0.1.0"
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

