//! Defines the basic `Evaluator` type. This is a struct that contains the particle Space and define the
//! underlying Greens function.

use super::particle_container::ParticleContainerAccessor;
use super::RealType;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num::complex::Complex;

/// Definition of allowed Kernel types.
pub enum KernelType<T: RealType> {
    Laplace,
    Helmholtz(Complex<T>),
    ModifiedHelmholtz(T),
}

pub enum ThreadingType {
    Parallel,
    Serial,
}

/// This type defines an Evaluator consisting of a
/// `ParticleSpace` and a `KernelType`.
pub struct RealDirectEvaluator<P: ParticleContainerAccessor> {
    kernel_type: KernelType<P::FloatingPointType>,
    particle_container: P,
}

pub trait RealDirectEvaluatorAccessor {
    type FloatingPointType: RealType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType<Self::FloatingPointType>;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType>;

    /// Assemble the kernel matrix in place
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    );

    /// Assemble the kernel matrix and return it
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::FloatingPointType>;
}

impl<P: ParticleContainerAccessor> RealDirectEvaluatorAccessor for RealDirectEvaluator<P> {
    type FloatingPointType = P::FloatingPointType;

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

    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        match threading_type {
            parallel => Zip::from(self.targets().columns())
                .and(result.rows_mut())
                .par_for_each(|target, mut result_row| {
                    laplace_kernel(&target, &self.sources(), &mut result_row)
                }),

            serial => Zip::from(targets.columns())
                .and(result.rows_mut())
                .for_each(|target, mut result_row| {
                    laplace_kernel(&target, &sources, &mut result_row)
                }),
        }
    }
}
