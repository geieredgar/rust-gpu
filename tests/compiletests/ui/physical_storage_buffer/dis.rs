#![crate_name = "physical_storage_buffer_dis"]

// Pins what accessing a device address emits: the capability and the
// `PhysicalStorageBuffer64` addressing model, neither of which the shader asks
// for, and the `OpConvertUToPtr` + `OpLoad`/`OpStore` the custom instructions
// become.
//
// The `Aligned` operand on both is the point of the entry disassembly: Vulkan
// requires one on every access through a physical pointer, and it is the one part
// of this that no later pass would add back.
//
// The load and the store share a pointee, so the single `OpTypePointer
// PhysicalStorageBuffer %uint` in the globals is also what says the pointer types
// are synthesized once rather than per access.

// build-pass
// compile-flags: -C target-feature=+Int64 -C llvm-args=--disassemble-globals -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "\n\W*OpSource .*" -> ""

use spirv_std::physical_storage_buffer::DevicePtr;
use spirv_std::spirv;

// Deliberately free of arithmetic, including `add`/`write_index`: anything that
// reaches `core`'s integer code puts an `OpString` naming a path inside the
// toolchain into the disassembly, which would pin this expectation to one
// machine.
#[spirv(fragment)]
pub fn main(#[spirv(flat)] address: u64, output: &mut u32) {
    let counter = DevicePtr::<u32>::from_bits(address);
    // SAFETY: the host put a `u32` at this address.
    let value = unsafe { counter.read() };
    unsafe { counter.write(value) };
    *output = value;
}
