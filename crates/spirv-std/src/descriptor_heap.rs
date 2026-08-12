//! Descriptor heaps, from `SPV_EXT_descriptor_heap`.
//!
//! A descriptor heap is the implementation's own array of descriptors, reached
//! through a built-in rather than bound to a set. Instead of declaring a binding
//! per resource, a shader indexes the heap:
//!
//! ```no_run
//! # #![cfg_attr(target_arch = "spirv", no_std)]
//! # #![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
//! use spirv_std::descriptor_heap::{ResourceHeap, SamplerHeap};
//! use spirv_std::image::Image2d;
//! use spirv_std::{Sampler, spirv};
//! use spirv_std::glam::{Vec2, Vec4};
//!
//! #[spirv(fragment)]
//! pub fn main(#[spirv(flat)] texture: u32, uv: Vec2, output: &mut Vec4) {
//!     let image: Image2d = unsafe { ResourceHeap::get(texture) };
//!     let sampler: Sampler = unsafe { SamplerHeap::get(0) };
//!     *output = image.sample(sampler, uv);
//! }
//! ```
//!
//! There is nothing in the entry point's signature: the heaps are per-device, not
//! per-draw, so they are reached like any other built-in rather than passed in.
//! They need no `descriptor_set` or `binding` — that is the point of the
//! extension — and the capability and `OpExtension` are declared for you as soon
//! as a heap is used.
//!
//! # Which heap holds what
//!
//! Vulkan splits descriptors into two heaps, and so does this: samplers come from
//! the [`SamplerHeap`] and everything else from the [`ResourceHeap`]. Asking a
//! heap for the wrong kind of descriptor does not compile.
//!
//! # Indices are not checked
//!
//! Nothing here knows how large a heap is; the host decides that when it fills
//! one. An out-of-range index is undefined behaviour, exactly as with descriptor
//! indexing, which is why [`ResourceHeap::get`] and [`SamplerHeap::get`] are
//! `unsafe`.

use crate::image::Image;
use crate::ray_tracing::AccelerationStructure;
use crate::sampler::Sampler;

/// Loads a descriptor from the resource heap.
///
/// `rustc_codegen_spirv` replaces calls to this with a custom instruction, which
/// its linker expands into an `OpUntypedAccessChainKHR` into the heap built-in
/// followed by an `OpLoad` of `T`.
#[spirv(resource_heap_load_intrinsic)]
// Inlining this would dissolve the call the codegen looks for.
#[inline(never)]
#[spirv_std_macros::gpu_only]
unsafe fn resource_heap_load_intrinsic<T>(index: u32) -> T {
    // Not reachable on the GPU: the codegen replaces every call to this. The body
    // exists so that a failure to do so is loud rather than silent.
    unsafe {
        let _ = index;
        core::hint::unreachable_unchecked()
    }
}

/// Loads a sampler from the sampler heap. See [`resource_heap_load_intrinsic`].
#[spirv(sampler_heap_load_intrinsic)]
#[inline(never)]
#[spirv_std_macros::gpu_only]
unsafe fn sampler_heap_load_intrinsic(index: u32) -> Sampler {
    unsafe {
        let _ = index;
        core::hint::unreachable_unchecked()
    }
}

/// A descriptor that lives in the [`ResourceHeap`].
///
/// # Safety
///
/// The implementing type's SPIR-V type must be one the resource heap can hold: an
/// image, a combined sampled image, or an acceleration structure. Loading a
/// descriptor of any other type out of a heap is not valid SPIR-V.
pub unsafe trait ResourceDescriptor: Copy {}

// SAFETY: `Image` is `OpTypeImage`, or `OpTypeSampledImage` when combined.
unsafe impl<
    SampledType: crate::image::SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
> ResourceDescriptor
    for Image<SampledType, DIM, DEPTH, ARRAYED, MULTISAMPLED, SAMPLED, FORMAT, COMPONENTS>
{
}

// SAFETY: `AccelerationStructure` is `OpTypeAccelerationStructureKHR`.
unsafe impl ResourceDescriptor for AccelerationStructure {}

/// The resource descriptor heap: images, combined sampled images and acceleration
/// structures.
///
/// Never constructed — it exists to name the heap that [`ResourceHeap::get`]
/// reads from.
pub enum ResourceHeap {}

impl ResourceHeap {
    /// The descriptor at `index`.
    ///
    /// # Safety
    ///
    /// `index` must be one the host filled in with a descriptor of type `T`. The
    /// heap's length is not known here, and neither is what the host put where —
    /// reading past the end, or reading a texture as a different kind of texture,
    /// is undefined behaviour.
    #[spirv_std_macros::gpu_only]
    #[inline]
    pub unsafe fn get<T: ResourceDescriptor>(index: u32) -> T {
        unsafe { resource_heap_load_intrinsic(index) }
    }
}

/// The sampler descriptor heap.
///
/// Separate from the [`ResourceHeap`] because Vulkan keeps samplers in a heap of
/// their own.
pub enum SamplerHeap {}

impl SamplerHeap {
    /// The sampler at `index`.
    ///
    /// # Safety
    ///
    /// As [`ResourceHeap::get`]: `index` must name a sampler the host put in the
    /// sampler heap.
    #[spirv_std_macros::gpu_only]
    #[inline]
    pub unsafe fn get(index: u32) -> Sampler {
        unsafe { sampler_heap_load_intrinsic(index) }
    }
}
