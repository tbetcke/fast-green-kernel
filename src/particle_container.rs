use crate::RealType;
use ndarray::{Array2, ArrayView2};

/// The basic data structure of this library for sources and targets
/// that are owned by the structure.
pub struct ParticleContainer<T: RealType> {
    sources: Array2<T>,
    targets: Array2<T>,
}

impl<T: RealType> ParticleContainer<T> {
    // Create a new ParticleContainerView
    pub(crate) fn new(sources: Array2<T>, targets: Array2<T>) -> ParticleContainer<T> {
        ParticleContainer { sources, targets }
    }
}

// The basic data structure of this library for sources and targets
// that are not owned by the structure.
pub struct ParticleContainerView<'a, T: RealType> {
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
}

impl<'a, T: RealType> ParticleContainerView<'a, T> {
    // Create a new ParticleContainerView
    pub(crate) fn new(
        sources: ArrayView2<'a, T>,
        targets: ArrayView2<'a, T>,
    ) -> ParticleContainerView<'a, T> {
        ParticleContainerView { sources, targets }
    }
}

// This traits describes any type that provides an array of sources and
// an array of targets.
pub trait ParticleContainerAccessor {
    type FloatingPointType: RealType;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType>;
}

impl<T: RealType> ParticleContainerAccessor for ParticleContainer<T> {
    type FloatingPointType = T;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}

impl<'a, T: RealType> ParticleContainerAccessor for ParticleContainerView<'a, T> {
    type FloatingPointType = T;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}

// /// This is the basic type describing a configuration of
// /// and targets with associated methods.
// pub struct ParticleSpace<P: ParticleContainerAccessor> {
//     particle_container: P,
// }

// impl<P: ParticleContainerAccessor> ParticleSpace<P> {
//     /// Return a non-owning representation of the sources.
//     pub fn sources(&self) -> ArrayView2<P::FloatingPointType> {
//         self.particle_container.sources()
//     }

//     /// Return a non-owning representation of the targets.
//     pub fn targets(&self) -> ArrayView2<P::FloatingPointType> {
//         self.particle_container.targets()
//     }
// }

// impl<T: SupportedType> ParticleSpace<ParticleContainer<T>> {
//     /// Create a new particle space by transferring ownership of sources and targets.
//     pub fn from_arrays(
//         sources: Array2<T>,
//         targets: Array2<T>,
//     ) -> ParticleSpace<ParticleContainer<T>> {
//         use ndarray::Axis;

//         if sources.len_of(Axis(0)) != 3 {
//             panic!(
//                 "`from_array_views: First dimension of sources is {} != 3",
//                 sources.len_of(Axis(0))
//             );
//         }

//         if targets.len_of(Axis(0)) != 3 {
//             panic!(
//                 "`from_array_views: First dimension of sources is {} != 3",
//                 sources.len_of(Axis(0))
//             );
//         }

//         ParticleSpace::<ParticleContainer<T>> {
//             particle_container: ParticleContainer { sources, targets },
//         }
//     }
// }

// impl<'a, T: SupportedType> ParticleSpace<ParticleContainer<T>> {
//     /// Create new particle space from a view onto sources and targets.
//     pub fn from_array_views(
//         sources: ArrayView2<'a, T>,
//         targets: ArrayView2<'a, T>,
//     ) -> ParticleSpace<ParticleContainerView<'a, T>> {
//         use ndarray::Axis;

//         if sources.len_of(Axis(0)) != 3 {
//             panic!(
//                 "`from_array_views: First dimension of sources is {} != 3",
//                 sources.len_of(Axis(0))
//             );
//         }

//         if targets.len_of(Axis(0)) != 3 {
//             panic!(
//                 "`from_array_views: First dimension of sources is {} != 3",
//                 sources.len_of(Axis(0))
//             );
//         }

//         ParticleSpace::<ParticleContainerView<'a, T>> {
//             particle_container: ParticleContainerView::<'a, T> { sources, targets },
//         }
//     }
// }
