use crate::{SupportedType, ParticleSpace, ParticleSpaceView};
use ndarray::{Array2, ArrayView2};

impl<T: SupportedType> ParticleSpace<T> {
    /// Create a new `ParticleSpace` struct.
    pub fn new(sources: Array2<T>, targets: Array2<T>) -> ParticleSpace<T> {
        crate::ParticleSpace { sources, targets }
    }
}

impl<'a, T: crate::SupportedType> crate::ParticleSpaceView<'a, T> {
    /// Create a new `ParticleSpaceView` struct.
    pub fn new(sources: ArrayView2<'a, T>, targets: ArrayView2<'a, T>) -> ParticleSpaceView<'a, T> {
        ParticleSpaceView { sources, targets }
    }
}

/// Basic trait defining a particle space.
pub trait ParticleSpaceTrait {
    type FloatingPointType;

    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType>;
    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType>;
}

impl<T: SupportedType> ParticleSpaceTrait for ParticleSpace<T> {
    type FloatingPointType = T;

    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}

impl<T: SupportedType> ParticleSpaceTrait for ParticleSpaceView<'_, T> {
    type FloatingPointType = T;

    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}
