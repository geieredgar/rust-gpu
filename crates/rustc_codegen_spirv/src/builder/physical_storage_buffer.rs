//! Loads through raw device addresses, from `SPV_KHR_physical_storage_buffer`.
//!
//! Reading through a device address is an `OpConvertUToPtr` to a
//! `PhysicalStorageBuffer` pointer followed by an `OpLoad` carrying an `Aligned`
//! memory operand. Neither half survives the SPIR-T round-trip usefully — see
//! [`crate::custom_insts`] for why — so codegen emits one custom instruction and
//! [`crate::linker::physical_storage_buffer`] expands it once the module is back
//! to SPIR-V.

use super::Builder;
use crate::builder_spirv::SpirvValue;
use crate::custom_insts::CustomInst;
use crate::spirv_type::SpirvType;
use rspirv::dr::Operand;
use rspirv::spirv::Word;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Codegens `physical_storage_buffer_load::<T>(address) -> T`.
    ///
    /// The result type decides what is read, and its alignment decides the
    /// `Aligned` memory operand — Vulkan requires one on every access through a
    /// physical pointer, and only the Rust type knows what it should be.
    pub fn codegen_physical_storage_buffer_load_intrinsic(
        &mut self,
        result_type: Word,
        args: &[SpirvValue],
    ) -> SpirvValue {
        let [address] = args else {
            self.fatal(format!(
                "physical storage buffer load intrinsic should have 1 arg, it has {}",
                args.len()
            ));
        };

        // Checked here, where there is still a span and a Rust type to talk
        // about; by the time the linker expands this, neither is around.
        let ty = self.lookup_type(result_type);
        match ty {
            // A pointer read out of memory would be a physical pointer itself,
            // which nothing downstream of here is prepared to lay out — see the
            // `physical_size` hole noted in `SpirvType::physical_size`.
            SpirvType::Pointer { .. } => {
                self.struct_err("cannot load a pointer through a device address")
                    .with_note(
                        "physical pointers are not a type this backend can lay out; \
                         load the address as a `u64` and dereference it in turn",
                    )
                    .emit();
                return self.undef(result_type);
            }
            // These are descriptors, which live in a heap and are reached by
            // index — never by address.
            SpirvType::Image { .. }
            | SpirvType::SampledImage { .. }
            | SpirvType::Sampler
            | SpirvType::AccelerationStructureKhr
            | SpirvType::RayQueryKhr => {
                self.struct_err(format!(
                    "`{}` cannot be loaded through a device address",
                    ty.debug(result_type, self)
                ))
                .with_note("descriptors are reached through a descriptor heap, by index")
                .emit();
                return self.undef(result_type);
            }
            // A runtime array has no size, so there is nothing to load; an
            // address names one element, and the caller indexes it themselves.
            SpirvType::RuntimeArray { .. } => {
                self.struct_err("cannot load an unsized value through a device address")
                    .with_note("load one element at a time, advancing the address yourself")
                    .emit();
                return self.undef(result_type);
            }
            _ => {}
        }

        let alignment = ty.alignof(self).bytes() as u32;
        let alignment = self.constant_u32(self.span(), alignment).def(self);

        self.custom_inst(
            result_type,
            CustomInst::PhysicalStorageBufferLoad {
                address: Operand::IdRef(address.def(self)),
                alignment: Operand::IdRef(alignment),
            },
        )
    }
}
