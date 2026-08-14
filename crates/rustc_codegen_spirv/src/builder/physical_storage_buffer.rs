//! Accesses through raw device addresses, from `SPV_KHR_physical_storage_buffer`.
//!
//! Reading through a device address is an `OpConvertUToPtr` to a
//! `PhysicalStorageBuffer` pointer followed by an `OpLoad` carrying an `Aligned`
//! memory operand, and writing is the same conversion followed by an `OpStore`.
//! Neither half survives the SPIR-T round-trip usefully — see
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

        if !self.check_physical_access_type(result_type, "loaded") {
            return self.undef(result_type);
        }

        let alignment = self.physical_access_alignment(result_type);
        self.custom_inst(
            result_type,
            CustomInst::PhysicalStorageBufferLoad {
                address: Operand::IdRef(address.def(self)),
                alignment: Operand::IdRef(alignment),
            },
        )
    }

    /// Codegens `physical_storage_buffer_store::<T>(address, value)`.
    ///
    /// Carries no result type, since an `OpStore` produces nothing: the pointee
    /// type is the type of `value`, which the linker looks up.
    pub fn codegen_physical_storage_buffer_store_intrinsic(&mut self, args: &[SpirvValue]) {
        let [address, value] = args else {
            self.fatal(format!(
                "physical storage buffer store intrinsic should have 2 args, it has {}",
                args.len()
            ));
        };

        if !self.check_physical_access_type(value.ty, "stored") {
            return;
        }

        let alignment = self.physical_access_alignment(value.ty);
        // FIXME(eddyb) this should be cached more efficiently.
        let void_ty = SpirvType::Void.def(self.span(), self);
        self.custom_inst(
            void_ty,
            CustomInst::PhysicalStorageBufferStore {
                address: Operand::IdRef(address.def(self)),
                alignment: Operand::IdRef(alignment),
                value: Operand::IdRef(value.def(self)),
            },
        );
    }

    /// The `Aligned` memory operand for an access to `ty`, as a `u32` constant.
    ///
    /// A constant rather than a literal because SPIR-T carries `OpExtInst`
    /// operands as values, so this is what survives to the linker.
    fn physical_access_alignment(&mut self, ty: Word) -> Word {
        let alignment = self.lookup_type(ty).alignof(self).bytes() as u32;
        self.constant_u32(self.span(), alignment).def(self)
    }

    /// Whether `ty` is something a device address can name.
    ///
    /// Checked here, where there is still a span and a Rust type to talk about;
    /// by the time the linker expands the access, neither is around. `verb` is
    /// how the message should describe what was being done to it.
    fn check_physical_access_type(&mut self, ty: Word, verb: &str) -> bool {
        match self.lookup_type(ty) {
            // A pointer in memory would be a physical pointer itself, which
            // nothing downstream of here is prepared to lay out — see the
            // `physical_size` hole noted in `SpirvType::physical_size`.
            SpirvType::Pointer { .. } => {
                self.struct_err(format!("a pointer cannot be {verb} through a device address"))
                    .with_note(
                        "physical pointers are not a type this backend can lay out; \
                         use a `u64` and dereference it in turn",
                    )
                    .emit();
                false
            }
            // These are descriptors, which live in a heap and are reached by
            // index — never by address.
            SpirvType::Image { .. }
            | SpirvType::SampledImage { .. }
            | SpirvType::Sampler
            | SpirvType::AccelerationStructureKhr
            | SpirvType::RayQueryKhr => {
                self.struct_err(format!(
                    "`{}` cannot be {verb} through a device address",
                    self.lookup_type(ty).debug(ty, self)
                ))
                .with_note("descriptors are reached through a descriptor heap, by index")
                .emit();
                false
            }
            // A runtime array has no size, so there is nothing to move; an
            // address names one element, and the caller indexes it themselves.
            SpirvType::RuntimeArray { .. } => {
                self.struct_err(format!(
                    "an unsized value cannot be {verb} through a device address"
                ))
                .with_note("use one element at a time, advancing the address yourself")
                .emit();
                false
            }
            _ => true,
        }
    }
}
