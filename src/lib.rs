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

use ndarray::{Array2, ArrayView2};
use num_traits;

/// The basic data structure of this library for sources and targets
/// that are owned by the structure.
pub struct ParticleSpaceContainer<T: SupportedType> {
    sources: Array2<T>,
    targets: Array2<T>,
}


// The basic data structure of this library for sources and targets
// that are not owned by the structure.
pub struct ParticleSpaceContainerView<'a, T: SupportedType> {
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
}

/// This trait specifies the required floating point properties.
/// Currently, we support f32 and f64. The actual type is contained
/// in the trait `BasePrecision`.
pub trait SupportedType:
    num_traits::Float
    + num_traits::FloatConst
    + num_traits::NumAssignOps
    + std::marker::Send
    + std::marker::Sync
    + BasePrecision // This trait specifies the bound f32 or f64
{
    type FloatingType;
}

impl<T> SupportedType for T
where
    T: num_traits::Float,
    T: num_traits::FloatConst,
    T: num_traits::NumAssignOps,
    T: std::marker::Send,
    T: std::marker::Sync,
    T: BasePrecision,
{
    type FloatingType = <T as BasePrecision>::FloatingType;
}

pub trait BasePrecision {
    type FloatingType;
}

impl BasePrecision for f32 {
    type FloatingType = f32;
}

impl BasePrecision for f64 {
    type FloatingType = f64;
}

pub mod base;
