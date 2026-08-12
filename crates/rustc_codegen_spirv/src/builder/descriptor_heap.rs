//! Descriptor heap loads, from `SPV_EXT_descriptor_heap`.
//!
//! Reading a descriptor out of a heap is an `OpUntypedAccessChainKHR` into the
//! heap built-in followed by an `OpLoad`, over an `OpTypeRuntimeArray` strided by
//! `OpConstantSizeOfEXT` (only the implementation knows how large a descriptor
//! is). None of that survives SPIR-T — the access chain and the size constant
//! both take a *type* as an operand, and `ArrayStrideIdEXT` is an `OpDecorateId`,
//! none of which SPIR-T can represent — so codegen emits one custom instruction
//! instead, and [`crate::linker::descriptor_heap`] expands it into the real
//! sequence once the module is back to SPIR-V.

use super::Builder;
use crate::builder_spirv::SpirvValue;
use crate::custom_insts::{CustomInst, DescriptorHeap};
use crate::spirv_type::SpirvType;
use rspirv::dr::Operand;
use rspirv::spirv::Word;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Codegens `resource_heap_load::<T>(index) -> T` and its sampler twin.
    ///
    /// The result type decides which heap array is indexed, so `T` has to be a
    /// descriptor type, and one the given heap can hold.
    pub fn codegen_descriptor_heap_load_intrinsic(
        &mut self,
        heap: DescriptorHeap,
        result_type: Word,
        args: &[SpirvValue],
    ) -> SpirvValue {
        let [index] = args else {
            self.fatal(format!(
                "descriptor heap load intrinsic should have 1 arg, it has {}",
                args.len()
            ));
        };

        // Checked here, where there is still a span and a Rust type to talk
        // about; by the time the linker expands this, neither is around.
        let ty = self.lookup_type(result_type);
        let fits = match heap {
            DescriptorHeap::Resource => matches!(
                ty,
                SpirvType::Image { .. }
                    | SpirvType::SampledImage { .. }
                    | SpirvType::AccelerationStructureKhr
            ),
            DescriptorHeap::Sampler => matches!(ty, SpirvType::Sampler),
        };
        if !fits {
            self.struct_err(format!(
                "`{}` cannot be loaded from the {} descriptor heap",
                ty.debug(result_type, self),
                match heap {
                    DescriptorHeap::Resource => "resource",
                    DescriptorHeap::Sampler => "sampler",
                },
            ))
            .with_note(match heap {
                DescriptorHeap::Resource => {
                    "the resource heap holds images, combined sampled images \
                     and acceleration structures"
                }
                DescriptorHeap::Sampler => "the sampler heap holds samplers",
            })
            .emit();
            return self.undef(result_type);
        }

        let heap = self.constant_u32(self.span(), heap as u32).def(self);
        self.custom_inst(
            result_type,
            CustomInst::DescriptorHeapLoad {
                heap: Operand::IdRef(heap),
                index: Operand::IdRef(index.def(self)),
            },
        )
    }
}
