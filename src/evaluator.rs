//! Defines the basic `Evaluator` type. This is a struct that contains the particle Space and define the
//! underlying Greens function.

use super::particle_container::ParticleContainerAccessor;
use super::{RealType, ResultType};
use ndarray::ArrayView2;
use num::complex::Complex;

/// Definition of allowed Kernel types.
pub enum KernelType<T: RealType> {
    Laplace,
    Helmholtz(Complex<T>),
    ModifiedHelmholtz(T),
}

/// This type defines an Evaluator consisting of a
/// `ParticleSpace` and a `KernelType`.
pub struct EvaluatorBase<P: ParticleContainerAccessor, R: ResultType> {
    phantom: std::marker::PhantomData<R>,
    kernel_type: KernelType<P::FloatingPointType>,
    particle_container: P,
}

pub trait Evaluator {
    type FloatingPointType: RealType;
    type ResultType: ResultType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType<Self::FloatingPointType>;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType>;
}

impl<P: ParticleContainerAccessor, R: ResultType> Evaluator for EvaluatorBase<P, R> {
    type FloatingPointType = P::FloatingPointType;
    type ResultType = R;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType<Self::FloatingPointType> {
        &self.kernel_type
    }

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.particle_container.sources()
    }
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.particle_container.targets()
    }
}

pub mod direct;
