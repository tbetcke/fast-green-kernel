//! Defines the basic `Evaluator` type. This is a struct that contains the particle Space and define the
//! underlying Greens function.

use super::kernels::EvalMode;
use super::particle_container::{
    ParticleContainer, ParticleContainerAccessor, ParticleContainerView,
};
use super::RealType;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use num::complex::Complex;

/// Definition of allowed Kernel types.
pub enum KernelType {
    Laplace,
    Helmholtz(Complex<f64>),
    ModifiedHelmholtz(f64),
}

pub enum ThreadingType {
    Parallel,
    Serial,
}

/// Make a Laplace evaluator from references to the data.
pub fn make_laplace_evaluator<'a, T: RealType>(
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
) -> DirectEvaluator<ParticleContainerView<'a, T>, T> {
    DirectEvaluator::<ParticleContainerView<'a, T>, T> {
        kernel_type: KernelType::Laplace,
        particle_container: ParticleContainerView::new(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}

/// Make a Laplace evaluator by taking ownership.
pub fn make_laplace_evaluator_owned<T: RealType>(
    sources: Array2<T>,
    targets: Array2<T>,
) -> DirectEvaluator<ParticleContainer<T>, T> {
    DirectEvaluator::<ParticleContainer<T>, T> {
        kernel_type: KernelType::Laplace,
        particle_container: ParticleContainer::new(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}

/// Make a Helmholtz evaluator from references to the data.
pub fn make_helmholtz_evaluator<'a, T: RealType>(
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
    wavenumber: Complex<f64>,
) -> DirectEvaluator<ParticleContainerView<'a, T>, num::complex::Complex<T>> {
    DirectEvaluator::<ParticleContainerView<'a, T>, num::complex::Complex<T>> {
        kernel_type: KernelType::Helmholtz(wavenumber),
        particle_container: ParticleContainerView::new(sources, targets),
        _marker: std::marker::PhantomData::<num::complex::Complex<T>>,
    }
}

/// Make a Helmholtz evaluator by taking ownership.
pub fn make_helmholtz_evaluator_owned<T: RealType>(
    sources: Array2<T>,
    targets: Array2<T>,
    wavenumber: Complex<f64>,
) -> DirectEvaluator<ParticleContainer<T>, Complex<T>> {
    DirectEvaluator::<ParticleContainer<T>, Complex<T>> {
        kernel_type: KernelType::Helmholtz(wavenumber),
        particle_container: ParticleContainer::new(sources, targets),
        _marker: std::marker::PhantomData::<Complex<T>>,
    }
}

/// This type defines an Evaluator consisting of a
/// `ParticleSpace` and a `KernelType`. The generic
pub struct DirectEvaluator<P: ParticleContainerAccessor, R> {
    kernel_type: KernelType,
    particle_container: P,
    _marker: std::marker::PhantomData<R>,
}

pub trait DirectEvaluatorAccessor {
    type FloatingPointType: RealType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType>;

    // Return number of sources
    fn nsources(&self) -> usize;

    // Return number of targets;
    fn ntargets(&self) -> usize;
}

pub trait RealDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place.
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    );

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView1<Self::FloatingPointType>,
        result: &mut ArrayViewMut2<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    );

    /// Assemble the kernel matrix and return it.
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::FloatingPointType>;

    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView1<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array2<Self::FloatingPointType>;
}

pub trait ComplexDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: &mut ArrayViewMut2<num::complex::Complex<Self::FloatingPointType>>,
        threading_type: ThreadingType,
    );

    /// Assemble the kernel matrix and return it
    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::FloatingPointType>>;
}

impl<P: ParticleContainerAccessor, R> DirectEvaluatorAccessor for DirectEvaluator<P, R> {
    type FloatingPointType = P::FloatingPointType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType {
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

    // Return number of sources.
    fn nsources(&self) -> usize {
        self.sources().len_of(Axis(1))
    }

    fn ntargets(&self) -> usize {
        self.targets().len_of(Axis(1))
    }
}

impl<P: ParticleContainerAccessor> RealDirectEvaluator
    for DirectEvaluator<P, P::FloatingPointType>
{
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    ) {
        match self.kernel_type {
            KernelType::Laplace => assemble_in_place_impl_laplace::<Self::FloatingPointType>(
                self.sources(),
                self.targets(),
                result,
                threading_type,
            ),
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    /// Assemble the kernel matrix and return it
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::FloatingPointType> {
        let mut result =
            Array2::<Self::FloatingPointType>::zeros((self.ntargets(), self.nsources()));

        self.assemble_in_place(result.view_mut(), threading_type);
        result
    }

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView1<Self::FloatingPointType>,
        result: &mut ArrayViewMut2<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) {
        evaluate_in_place_impl_laplace(
            self.sources(),
            self.targets(),
            charges,
            result,
            eval_mode,
            threading_type,
        );
    }
    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView1<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array2<Self::FloatingPointType> {
        let chunks = match eval_mode {
            EvalMode::Value => 1,
            EvalMode::ValueGrad => 4,
        };

        let mut result = Array2::<Self::FloatingPointType>::zeros((self.ntargets(), chunks));
        evaluate_in_place_impl_laplace(
            self.sources(),
            self.targets(),
            charges,
            &mut result.view_mut(),
            eval_mode,
            threading_type,
        );
        result
    }
}

impl<P: ParticleContainerAccessor> ComplexDirectEvaluator
    for DirectEvaluator<P, num::complex::Complex<P::FloatingPointType>>
{
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: &mut ArrayViewMut2<num::complex::Complex<Self::FloatingPointType>>,
        threading_type: ThreadingType,
    ) {
        match self.kernel_type {
            KernelType::Helmholtz(wavenumber) => {
                assemble_in_place_impl_helmholtz::<Self::FloatingPointType>(
                    self.sources(),
                    self.targets(),
                    result,
                    wavenumber,
                    threading_type,
                )
            }
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    /// Assemble the kernel matrix and return it
    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::FloatingPointType>> {
        let mut result = Array2::<num::complex::Complex<Self::FloatingPointType>>::zeros((
            self.nsources(),
            self.ntargets(),
        ));

        self.assemble_in_place(&mut result.view_mut(), threading_type);
        result
    }
}

/// Implementation of assembler function for Laplace kernels.
fn assemble_in_place_impl_laplace<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    mut result: ArrayViewMut2<T>,
    threading_type: ThreadingType,
) {
    use crate::kernels::laplace_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, result_row| {
                let mut tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot convert to 2-dimensional array.");
                laplace_kernel(target, sources, &mut tmp, &EvalMode::Value);
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, result_row| {
                let mut tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot conver to 2-dimensional array.");
                laplace_kernel(target, sources, &mut tmp, &EvalMode::Value);
            }),
    }
}

/// Implementation of assembler function for Helmholtz.
fn assemble_in_place_impl_helmholtz<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    result: &mut ArrayViewMut2<num::complex::Complex<T>>,
    wavenumber: num::complex::Complex<f64>,
    threading_type: ThreadingType,
) {
    use crate::kernels::helmholtz_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, result_row| {
                let mut tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot convert to 2-dimensional array.");
                helmholtz_kernel(target, sources, &mut tmp, wavenumber, &EvalMode::Value);
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, result_row| {
                let mut tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot conver to 2-dimensional array.");
                helmholtz_kernel(target, sources, &mut tmp, wavenumber, &EvalMode::Value);
            }),
    }
}

/// Implementation of assembler function for Laplace kernels.
fn evaluate_in_place_impl_laplace<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    charges: ArrayView1<T>,
    result: &mut ArrayViewMut2<T>,
    eval_mode: &EvalMode,
    threading_type: ThreadingType,
) {
    use crate::kernels::laplace_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    let chunks = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, mut result_row| {
                let mut tmp = Array2::<T>::zeros((chunks, nsources));
                laplace_kernel(target, sources, &mut tmp.view_mut(), eval_mode);
                Zip::from(tmp.rows())
                    .and(result_row.view_mut())
                    .for_each(|tmp_row, result_elem| {
                        Zip::from(tmp_row.view()).and(charges.view()).for_each(
                            |tmp_value, charge_value| *result_elem += *tmp_value * *charge_value,
                        )
                    })
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, mut result_row| {
                let mut tmp = Array2::<T>::zeros((chunks, nsources));
                laplace_kernel(target, sources, &mut tmp.view_mut(), eval_mode);
                Zip::from(tmp.rows())
                    .and(result_row.view_mut())
                    .for_each(|tmp_row, result_elem| {
                        Zip::from(tmp_row.view()).and(charges.view()).for_each(
                            |tmp_value, charge_value| *result_elem += *tmp_value * *charge_value,
                        )
                    })
            }),
    };
}
