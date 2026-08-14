//! Expands `CustomInst::PhysicalStorageBufferLoad` into
//! `SPV_KHR_physical_storage_buffer`.
//!
//! This runs *after* `spirt::spv::lift`, on the SPIR-V the SPIR-T pipeline
//! produced, because a `PhysicalStorageBuffer` pointer cannot exist before then:
//! the whole codegen assumes the `Logical` addressing model, in which a pointer
//! is a value the implementation understands rather than a number, and
//! `SpirvType::Pointer` accordingly carries no storage class to vary.
//!
//! Each custom instruction becomes:
//!
//! ```spirv
//! %ptr = OpConvertUToPtr %ptr_type %address
//! %value = OpLoad %type %ptr Aligned <alignment>
//! ```
//!
//! with the pointer types, the capability and the module's addressing model all
//! synthesized here, once per module.

use crate::custom_insts::{self, CustomInst, CustomOp};
use rspirv::dr::{Block, Instruction, Module, Operand};
use rspirv::spirv::{AddressingModel, Capability, MemoryAccess, Op, StorageClass, Word};
use rustc_data_structures::fx::FxHashMap;

/// One physical load found in a function body.
struct Load {
    /// The `OpExtInst`'s own result ID, which the `OpLoad` inherits so that
    /// everything already referring to it keeps working.
    result_id: Word,
    /// The type being loaded.
    loaded_type: Word,
    address_id: Word,
    /// The `Aligned` memory operand, in bytes.
    alignment: u32,
}

/// Everything synthesized for a module, created on first use.
struct Pointers {
    /// `OpTypePointer PhysicalStorageBuffer` per pointee type.
    types: FxHashMap<Word, Word>,
    /// The fresh ID of the `OpConvertUToPtr` feeding each load, keyed by the
    /// load's own result ID (which the `OpLoad` keeps).
    convert_ids: FxHashMap<Word, Word>,
}

/// Replaces every `CustomInst::PhysicalStorageBufferLoad` with the real thing.
///
/// Does nothing to a module that has none, which is all but a few.
pub fn expand_physical_storage_buffer_loads(module: &mut Module) {
    let Some(custom_ext_inst_set) = module
        .ext_inst_imports
        .iter()
        .find(|inst| {
            inst.operands[0]
                .unwrap_literal_string()
                .starts_with(custom_insts::CUSTOM_EXT_INST_SET_PREFIX)
        })
        .and_then(|inst| inst.result_id)
    else {
        return;
    };

    // Collected first, so that the pass can bail out without touching a module
    // that only uses the custom set for something else.
    let mut loads: Vec<Vec<Vec<Option<Load>>>> = vec![];
    let mut any = false;
    for func in &module.functions {
        let mut per_block = vec![];
        for block in &func.blocks {
            let mut per_inst = vec![];
            for inst in &block.instructions {
                let load = decode_load(module, custom_ext_inst_set, inst);
                any |= load.is_some();
                per_inst.push(load);
            }
            per_block.push(per_inst);
        }
        loads.push(per_block);
    }
    if !any {
        return;
    }

    let mut ids = IdAllocator::new(module);
    let mut pointers = Pointers {
        types: FxHashMap::default(),
        convert_ids: FxHashMap::default(),
    };

    // The pointer types have to exist before the conversions that name them, and
    // `pointer_type_for` appends to `types_global_values`, so they are all
    // created up-front rather than while rewriting the bodies.
    for per_block in &loads {
        for per_inst in per_block {
            for load in per_inst.iter().flatten() {
                pointers.pointer_type_for(module, &mut ids, load.loaded_type);
                let ptr_id = ids.alloc();
                pointers.convert_ids.insert(load.result_id, ptr_id);
            }
        }
    }

    for (func, per_block) in module.functions.iter_mut().zip(&loads) {
        for (block, per_inst) in func.blocks.iter_mut().zip(per_block) {
            rewrite_block(block, per_inst, &pointers);
        }
    }

    declare_capability(module);
    set_addressing_model(module);
    remove_unused_custom_ext_inst_set(module, custom_ext_inst_set);

    module.header.as_mut().unwrap().bound = ids.next;
}

/// Decodes one instruction, if it is a physical load.
fn decode_load(module: &Module, custom_ext_inst_set: Word, inst: &Instruction) -> Option<Load> {
    if inst.class.opcode != Op::ExtInst
        || inst.operands[0].unwrap_id_ref() != custom_ext_inst_set
        || CustomOp::decode_from_ext_inst(inst) != CustomOp::PhysicalStorageBufferLoad
    {
        return None;
    }

    let CustomInst::PhysicalStorageBufferLoad { address, alignment } = CustomInst::decode(inst)
    else {
        unreachable!("just decoded as `PhysicalStorageBufferLoad`");
    };

    // The alignment is a `u32` constant, because SPIR-T carries `OpExtInst`
    // operands as values - so it has to be read back out of the module here.
    let alignment = constant_u32(module, alignment.unwrap_id_ref())
        .expect("`PhysicalStorageBufferLoad`'s alignment operand must be a `u32` constant");

    Some(Load {
        result_id: inst.result_id.unwrap(),
        loaded_type: inst.result_type.unwrap(),
        address_id: address.unwrap_id_ref(),
        alignment,
    })
}

/// Replaces the physical loads in one block, leaving everything else alone.
fn rewrite_block(block: &mut Block, per_inst: &[Option<Load>], pointers: &Pointers) {
    if per_inst.iter().all(Option::is_none) {
        return;
    }

    let instructions = std::mem::take(&mut block.instructions);
    for (inst, load) in instructions.into_iter().zip(per_inst) {
        let Some(load) = load else {
            block.instructions.push(inst);
            continue;
        };

        // The conversion gets a fresh ID and the load keeps the original, so
        // that uses of the loaded value need no rewriting.
        let ptr_id = pointers.convert_ids[&load.result_id];
        block.instructions.push(Instruction::new(
            Op::ConvertUToPtr,
            Some(pointers.types[&load.loaded_type]),
            Some(ptr_id),
            vec![Operand::IdRef(load.address_id)],
        ));
        // The `Aligned` operand is not optional: Vulkan requires one on every
        // access through a physical pointer, since the implementation has no
        // other way to know what the address is aligned to.
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(load.loaded_type),
            Some(load.result_id),
            vec![
                Operand::IdRef(ptr_id),
                Operand::MemoryAccess(MemoryAccess::ALIGNED),
                Operand::LiteralBit32(load.alignment),
            ],
        ));
    }
}

/// Hands out IDs past everything the module already uses.
struct IdAllocator {
    next: Word,
}

impl IdAllocator {
    fn new(module: &Module) -> Self {
        Self {
            next: module.header.as_ref().unwrap().bound,
        }
    }

    fn alloc(&mut self) -> Word {
        let id = self.next;
        self.next += 1;
        id
    }
}

impl Pointers {
    /// The `OpTypePointer PhysicalStorageBuffer` for a pointee, created on first
    /// use.
    fn pointer_type_for(&mut self, module: &mut Module, ids: &mut IdAllocator, pointee: Word) {
        if self.types.contains_key(&pointee) {
            return;
        }

        let ptr_type = ids.alloc();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(ptr_type),
            vec![
                Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                Operand::IdRef(pointee),
            ],
        ));
        self.types.insert(pointee, ptr_type);
    }
}

/// The value of a `u32` `OpConstant`, if `id` names one.
fn constant_u32(module: &Module, id: Word) -> Option<u32> {
    module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(id) && inst.class.opcode == Op::Constant)
        .map(|inst| inst.operands[0].unwrap_literal_bit32())
}

/// Declares the capability, and the extension when the version needs it.
fn declare_capability(module: &mut Module) {
    let has_capability = module.capabilities.iter().any(|inst| {
        inst.operands[0].unwrap_capability() == Capability::PhysicalStorageBufferAddresses
    });
    if !has_capability {
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(
                Capability::PhysicalStorageBufferAddresses,
            )],
        ));
    }

    // Core since SPIR-V 1.5, so only older modules need to ask for it.
    let (major, minor) = module.header.as_ref().unwrap().version();
    if (major, minor) >= (1, 5) {
        return;
    }
    let has_extension = module
        .extensions
        .iter()
        .any(|inst| inst.operands[0].unwrap_literal_string() == "SPV_KHR_physical_storage_buffer");
    if !has_extension {
        module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(
                "SPV_KHR_physical_storage_buffer".into(),
            )],
        ));
    }
}

/// Switches the module to the `PhysicalStorageBuffer64` addressing model.
///
/// `BuilderSpirv::new` sets `Logical`, which is right for every module that does
/// not name an address; a module that does may not keep it, and there is no third
/// option — `PhysicalStorageBuffer64` is what `SPV_KHR_physical_storage_buffer`
/// defines, and it leaves every other storage class working as before.
fn set_addressing_model(module: &mut Module) {
    let memory_model = module
        .memory_model
        .as_mut()
        .expect("a module always has an `OpMemoryModel`");
    memory_model.operands[0] = Operand::AddressingModel(AddressingModel::PhysicalStorageBuffer64);
}

/// Drops the custom "extended instruction set" import, now that nothing uses it.
///
/// The linker refuses to emit a module that still imports it, so leaving an
/// unused import behind would turn this pass into an error.
fn remove_unused_custom_ext_inst_set(module: &mut Module, custom_ext_inst_set: Word) {
    let still_used = module.functions.iter().any(|func| {
        func.blocks.iter().any(|block| {
            block.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ExtInst
                    && inst.operands[0].unwrap_id_ref() == custom_ext_inst_set
            })
        })
    });
    if !still_used {
        module
            .ext_inst_imports
            .retain(|inst| inst.result_id != Some(custom_ext_inst_set));
    }
}
