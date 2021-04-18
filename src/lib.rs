#![crate_name = "fast_green_kernel"]

//! # fast-green-kernel
//!
//! `fast-green-kernel` is a library for the fast evaluation of Green's fct. kernels.
//! The library is currently very early days. Over time goals for the library are:
//! * Dense evaluation of the scalar Laplace and Helmholtz kernel
//! * Evaluation of both kernels with FMM implementations
//! * Fast HSS like compression methods for kernels
//! * Python interfaces for all routines.

//! At the moment the only thing the library does is the dense evaluation of kernels
//! together with corresponding Python interfaces.
//!

/// This trait determines the supported precision. It is implemented
/// for f32 and f64.
pub trait BasePrecision {
    type FloatingType;
}

/// This trait specifies the required floating point properties for real types.
/// Currently, we support f32 and f64.
pub trait RealType:
    num::traits::NumAssignOps
    + std::marker::Send
    + std::marker::Sync
    + num::traits::Float
    + num::traits::FloatConst
    + std::fmt::Display
{
}

// pub trait ResultType:
//     num::traits::Num + num::traits::NumAssignOps + std::marker::Send + std::marker::Sync
// {
// }

// use num::complex::Complex;

impl BasePrecision for f32 {
    type FloatingType = f32;
}

impl BasePrecision for f64 {
    type FloatingType = f64;
}

impl RealType for f32 {}
impl RealType for f64 {}

// impl ResultType for Complex<f32> {}
// impl ResultType for Complex<f64> {}
// impl ResultType for f32 {}
// impl ResultType for f64 {}

pub mod kernels;
pub mod direct_evaluator;
pub mod particle_container;
pub mod utilities;
pub mod c_api;
