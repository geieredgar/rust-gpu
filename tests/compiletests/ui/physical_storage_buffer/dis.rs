#![crate_name = "physical_storage_buffer_dis"]

// Pins what reading a device address emits: the capability and the
// `PhysicalStorageBuffer64` addressing model, neither of which the shader asks
// for, and the `OpConvertUToPtr` + `OpLoad` pair the custom instruction becomes.
//
// The `Aligned` operand on the load is the point of the entry disassembly: Vulkan
// requires one on every access through a physical pointer, and it is the one part
// of this that no later pass would add back.

// build-pass
// compile-flags: -C target-feature=+Int64 -C llvm-args=--disassemble-globals -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "\n\W*OpSource .*" -> ""

use spirv_std::physical_storage_buffer::DevicePtr;
use spirv_std::spirv;

#[spirv(fragment)]
pub fn main(#[spirv(flat)] address: u64, output: &mut u32) {
    let counters = DevicePtr::<u32>::from_bits(address);
    // SAFETY: the host put a `u32` at this address.
    *output = unsafe { counters.read() };
}
