// Reads a struct out of memory named by a raw device address, with no descriptor
// and no binding in sight — the whole point of `SPV_KHR_physical_storage_buffer`.
//
// The capability and the `PhysicalStorageBuffer64` addressing model are declared
// by the linker when an address is read, so this test deliberately passes no
// `-C target-feature=+PhysicalStorageBufferAddresses`.

// build-pass
// `Int64` is not declared by the linker the way the physical-storage-buffer
// capability is: an address is a `u64` in the shader's own types, so it is needed
// before anything this feature emits, and asking for it is the shader's job.
// compile-flags: -C target-feature=+Int64

use spirv_std::glam::Vec4;
use spirv_std::physical_storage_buffer::DevicePtr;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Instance {
    pub color: Vec4,
}

#[spirv(fragment)]
pub fn main(#[spirv(flat)] address: u64, #[spirv(flat)] index: u32, output: &mut Vec4) {
    let instances = DevicePtr::<Instance>::from_bits(address);
    // SAFETY: the host put an array of `Instance` at this address, and `index` is
    // in range — neither of which anything here can check.
    let instance = unsafe { instances.index(index as usize) };
    *output = instance.color;
}
