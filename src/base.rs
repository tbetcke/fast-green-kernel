use crate::{SupportedType, ParticleSpaceContainer, ParticleSpaceContainerView};
use ndarray::{Array2, ArrayView2};

impl<T: SupportedType> ParticleSpaceContainer<T> {
    /// Create a new `ParticleSpace` struct.
    pub fn new(sources: Array2<T>, targets: Array2<T>) -> ParticleSpaceContainer<T> {
        crate::ParticleSpaceContainer { sources, targets }
    }
}

impl<'a, T: crate::SupportedType> crate::ParticleSpaceContainerView<'a, T> {
    /// Create a new `ParticleSpaceView` struct.
    pub fn new(sources: ArrayView2<'a, T>, targets: ArrayView2<'a, T>) -> ParticleSpaceContainerView<'a, T> {
        ParticleSpaceContainerView { sources, targets }
    }
}

/// Basic trait defining a particle space.
pub trait ParticleSpaceTrait {
    type FloatingPointType;

    /// Get the sources.
    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Get the targets.
    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType>;
}

impl<T: SupportedType> ParticleSpaceTrait for ParticleSpaceContainer<T> {
    type FloatingPointType = T;

    /// Get the sources.
    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    /// Get the targets.
    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}

impl<T: SupportedType> ParticleSpaceTrait for ParticleSpaceContainerView<'_, T> {
    type FloatingPointType = T;

    /// Get the sources.
    fn get_sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.sources.view()
    }

    /// Get the targets.
    fn get_targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.targets.view()
    }
}

