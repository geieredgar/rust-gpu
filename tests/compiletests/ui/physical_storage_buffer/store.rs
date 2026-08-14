// Writes through a device address, and reads one back, so that a load and a
// store share a pointee type — the pointer type is synthesized once and used by
// both.

// build-pass
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
pub fn main(
    #[spirv(flat)] source: u64,
    #[spirv(flat)] destination: u64,
    #[spirv(flat)] index: u32,
    output: &mut Vec4,
) {
    let src = DevicePtr::<Instance>::from_bits(source);
    let dst = DevicePtr::<Instance>::from_bits(destination);

    // SAFETY: the host put arrays of `Instance` at both addresses, and `index`
    // is in range — neither of which anything here can check.
    let instance = unsafe { src.index(index as usize) };
    unsafe { dst.write_index(index as usize, instance) };

    *output = instance.color;
}
