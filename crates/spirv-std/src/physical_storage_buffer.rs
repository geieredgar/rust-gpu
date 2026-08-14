//! Reads and writes through raw device addresses, from
//! `SPV_KHR_physical_storage_buffer`.
//!
//! A buffer reached this way is not a descriptor and is not bound to anything:
//! the shader is handed a 64-bit address — usually through push data — and reads
//! or writes the memory at it. That is the whole interface.
//!
//! ```no_run
//! # #![cfg_attr(target_arch = "spirv", no_std)]
//! # #![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
//! use spirv_std::physical_storage_buffer::DevicePtr;
//! use spirv_std::spirv;
//! use spirv_std::glam::Vec4;
//!
//! #[repr(C)]
//! #[derive(Clone, Copy)]
//! pub struct Instance {
//!     pub color: Vec4,
//! }
//!
//! #[spirv(fragment)]
//! pub fn main(#[spirv(flat)] address: u64, output: &mut Vec4) {
//!     let instances = DevicePtr::<Instance>::from_bits(address);
//!     // SAFETY: the host put an array of `Instance` at this address.
//!     *output = unsafe { instances.add(2).read() }.color;
//! }
//! ```
//!
//! # `Int64` is the shader's to ask for
//!
//! An address is a `u64`, so a shader using this needs `OpCapability Int64` —
//! `-C target-feature=+Int64`. The capability and the addressing model that this
//! feature itself needs are declared by the linker when an address is read, but
//! `Int64` is not: it is about the shader's own types, and is required before
//! anything here is reached.
//!
//! # Nothing here is checked
//!
//! An address is a number the host chose; nothing on this side knows what is
//! there, how much of it there is, or whether it is still allocated. Reading the
//! wrong one is undefined behaviour, which is why every read is `unsafe` — the
//! same bargain the descriptor heaps make with their indices.
//!
//! # Layout has to agree
//!
//! `T` is read with the layout this compiler gives it, so it has to be the layout
//! the host wrote. Use `#[repr(C)]` and keep the type free of padding the two
//! sides could disagree about, exactly as for push data.
//!
//! # Alignment
//!
//! Vulkan requires every access through a physical pointer to state what the
//! address is aligned to, and the compiler states `align_of::<T>()`. So the
//! address must be at least that aligned, or the read is undefined behaviour —
//! `T`'s alignment is a promise the host keeps, not something the shader can
//! check.

use core::marker::PhantomData;

/// Loads a `T` from a device address.
///
/// `rustc_codegen_spirv` replaces calls to this with a custom instruction, which
/// its linker expands into an `OpConvertUToPtr` to a `PhysicalStorageBuffer`
/// pointer followed by an `OpLoad` carrying an `Aligned` memory operand.
#[spirv(physical_storage_buffer_load_intrinsic)]
// Inlining this would dissolve the call the codegen looks for.
#[inline(never)]
#[spirv_std_macros::gpu_only]
unsafe fn physical_storage_buffer_load_intrinsic<T>(address: u64) -> T {
    // Not reachable on the GPU: the codegen replaces every call to this. The body
    // exists so that a failure to do so is loud rather than silent.
    unsafe {
        let _ = address;
        core::hint::unreachable_unchecked()
    }
}

/// Stores a `T` to a device address.
///
/// `rustc_codegen_spirv` replaces calls to this with a custom instruction, which
/// its linker expands into an `OpConvertUToPtr` to a `PhysicalStorageBuffer`
/// pointer followed by an `OpStore` carrying an `Aligned` memory operand.
#[spirv(physical_storage_buffer_store_intrinsic)]
// Inlining this would dissolve the call the codegen looks for.
#[inline(never)]
#[spirv_std_macros::gpu_only]
unsafe fn physical_storage_buffer_store_intrinsic<T>(address: u64, value: T) {
    // Not reachable on the GPU: the codegen replaces every call to this. The body
    // exists so that a failure to do so is loud rather than silent.
    unsafe {
        let _ = (address, value);
        core::hint::unreachable_unchecked()
    }
}

/// A pointer to a `T` in memory the host named by address.
///
/// `repr(transparent)` over the address, so it can sit in a `#[repr(C)]` struct
/// shared with the host — usually push data — and mean the same thing on both
/// sides.
///
/// This is a *number*, not a borrow: it carries no lifetime and nothing keeps the
/// memory it names alive. See the module docs for what that costs.
#[repr(transparent)]
pub struct DevicePtr<T> {
    address: u64,
    _marker: PhantomData<fn() -> T>,
}

// Derived by hand: `#[derive]` would require `T: Copy`, and a pointer is `Copy`
// whatever it points at.
impl<T> Clone for DevicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DevicePtr<T> {}

impl<T> DevicePtr<T> {
    /// A pointer to the address in `bits`.
    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            address: bits,
            _marker: PhantomData,
        }
    }

    /// The raw address.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.address
    }

    /// The pointer `count` elements further along, as `<*const T>::add`.
    ///
    /// Wrapping arithmetic, since an address is only a number here; a pointer
    /// past the end of the allocation is not itself undefined, only reading it
    /// is.
    #[inline]
    pub const fn add(self, count: usize) -> Self {
        Self::from_bits(
            self.address
                .wrapping_add((count as u64).wrapping_mul(core::mem::size_of::<T>() as u64)),
        )
    }

    /// The pointer `count` elements back.
    #[inline]
    pub const fn sub(self, count: usize) -> Self {
        Self::from_bits(
            self.address
                .wrapping_sub((count as u64).wrapping_mul(core::mem::size_of::<T>() as u64)),
        )
    }

    /// Reads the `T` at this address.
    ///
    /// # Safety
    ///
    /// The address must name a live, readable `T` whose layout matches this one,
    /// aligned to at least `align_of::<T>()` — see the module docs. Nothing here
    /// checks any of it.
    #[inline]
    pub unsafe fn read(self) -> T {
        unsafe { physical_storage_buffer_load_intrinsic(self.address) }
    }

    /// Reads the `T` `index` elements along, as indexing an array would.
    ///
    /// # Safety
    ///
    /// As [`DevicePtr::read`], for the element at `index`.
    #[inline]
    pub unsafe fn index(self, index: usize) -> T {
        unsafe { self.add(index).read() }
    }

    /// Writes `value` to this address.
    ///
    /// # Safety
    ///
    /// The address must name live, writable memory laid out as `T` and aligned
    /// to at least `align_of::<T>()` — see the module docs. Nothing here checks
    /// any of it.
    ///
    /// Two invocations writing the same address race like any other GPU write:
    /// this carries no synchronization of its own, and a read of what another
    /// invocation wrote needs a barrier between the two.
    #[inline]
    pub unsafe fn write(self, value: T) {
        unsafe { physical_storage_buffer_store_intrinsic(self.address, value) }
    }

    /// Writes `value` `index` elements along, as indexing an array would.
    ///
    /// # Safety
    ///
    /// As [`DevicePtr::write`], for the element at `index`.
    #[inline]
    pub unsafe fn write_index(self, index: usize, value: T) {
        unsafe { self.add(index).write(value) }
    }
}
