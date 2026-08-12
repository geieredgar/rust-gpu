//! Expands `CustomInst::DescriptorHeapLoad` into `SPV_EXT_descriptor_heap`.
//!
//! This runs *after* `spirt::spv::lift`, on the SPIR-V the SPIR-T pipeline
//! produced, because none of what it emits can pass through SPIR-T:
//!
//! * `OpUntypedAccessChainKHR` takes a *type* as its "Base Type" operand, and
//!   `DataInstKind::SpvInst` inputs are values
//! * so does `OpConstantSizeOfEXT`, and `ConstKind::SpvInst` inputs are constants
//! * `ArrayStrideIdEXT` is an `OpDecorateId`, and `Attr::SpvAnnotation` holds no
//!   IDs at all
//! * `OpUntypedVariableKHR` is a module-scoped variable that SPIR-T's lowering
//!   does not recognize as one (it matches `OpVariable`)
//!
//! Each custom instruction becomes:
//!
//! ```spirv
//! %ptr = OpUntypedAccessChainKHR %uc_ptr %heap_array %heap %index
//! %descriptor = OpLoad %descriptor_type %ptr
//! ```
//!
//! with the heap variables, the per-descriptor-type arrays, their stride
//! constants and the capability all synthesized here, once per module.

use crate::custom_insts::{self, CustomInst, CustomOp, DescriptorHeap};
use rspirv::dr::{Block, Instruction, Module, Operand};
use rspirv::spirv::{Capability, Decoration, Op, StorageClass, Word};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

/// One heap load found in a function body.
struct Load {
    /// The `OpExtInst`'s own result ID, which the `OpLoad` inherits so that
    /// everything already referring to it keeps working.
    result_id: Word,
    /// The descriptor type being loaded.
    descriptor_type: Word,
    heap: DescriptorHeap,
    index_id: Word,
}

/// Everything synthesized for a module, created on first use.
struct Heaps {
    u32_type: Word,
    /// `OpTypeUntypedPointerKHR UniformConstant`.
    untyped_ptr_type: Word,
    /// The `OpUntypedVariableKHR` per heap.
    variables: FxHashMap<u32, Word>,
    /// `OpTypeRuntimeArray` per descriptor type, strided by its size.
    arrays: FxHashMap<Word, Word>,
    /// The fresh ID of the access chain feeding each load, keyed by the load's
    /// own result ID (which the `OpLoad` keeps).
    access_chain_ids: FxHashMap<Word, Word>,
}

/// Replaces every `CustomInst::DescriptorHeapLoad` with the real thing.
///
/// Does nothing to a module that has none, which is all but a few.
pub fn expand_descriptor_heap_loads(module: &mut Module) {
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
    let mut heaps = Heaps::new(module, &mut ids);

    // The arrays have to exist before the access chains that name them, and
    // `Heaps::array_for` appends to `types_global_values`, so they are all
    // created up-front rather than while rewriting the bodies.
    for per_block in &loads {
        for per_inst in per_block {
            for load in per_inst.iter().flatten() {
                heaps.array_for(module, &mut ids, load.descriptor_type);
                heaps.variable_for(module, &mut ids, load.heap);
                let ptr_id = ids.alloc();
                heaps.access_chain_ids.insert(load.result_id, ptr_id);
            }
        }
    }

    for (func, per_block) in module.functions.iter_mut().zip(&loads) {
        for (block, per_inst) in func.blocks.iter_mut().zip(per_block) {
            rewrite_block(block, per_inst, &heaps);
        }
    }

    declare_capability(module);
    add_heaps_to_entry_point_interfaces(module, &heaps);
    remove_unused_custom_ext_inst_set(module, custom_ext_inst_set);

    module.header.as_mut().unwrap().bound = ids.next;
}

/// Decodes one instruction, if it is a heap load.
fn decode_load(module: &Module, custom_ext_inst_set: Word, inst: &Instruction) -> Option<Load> {
    if inst.class.opcode != Op::ExtInst
        || inst.operands[0].unwrap_id_ref() != custom_ext_inst_set
        || CustomOp::decode_from_ext_inst(inst) != CustomOp::DescriptorHeapLoad
    {
        return None;
    }

    let CustomInst::DescriptorHeapLoad { heap, index } = CustomInst::decode(inst) else {
        unreachable!("just decoded as `DescriptorHeapLoad`");
    };

    // The heap is a `u32` constant, because SPIR-T carries `OpExtInst` operands
    // as values - so it has to be read back out of the module here.
    let heap = constant_u32(module, heap.unwrap_id_ref())
        .and_then(DescriptorHeap::decode)
        .expect("`DescriptorHeapLoad`'s heap operand must be a valid `u32` constant");

    Some(Load {
        result_id: inst.result_id.unwrap(),
        descriptor_type: inst.result_type.unwrap(),
        heap,
        index_id: index.unwrap_id_ref(),
    })
}

/// Replaces the heap loads in one block, leaving everything else alone.
fn rewrite_block(block: &mut Block, per_inst: &[Option<Load>], heaps: &Heaps) {
    if per_inst.iter().all(Option::is_none) {
        return;
    }

    let instructions = std::mem::take(&mut block.instructions);
    for (inst, load) in instructions.into_iter().zip(per_inst) {
        let Some(load) = load else {
            block.instructions.push(inst);
            continue;
        };

        // The access chain gets a fresh ID and the load keeps the original, so
        // that uses of the descriptor need no rewriting.
        let ptr_id = heaps.access_chain_ids[&load.result_id];
        block.instructions.push(Instruction::new(
            Op::UntypedAccessChainKHR,
            Some(heaps.untyped_ptr_type),
            Some(ptr_id),
            vec![
                Operand::IdRef(heaps.arrays[&load.descriptor_type]),
                Operand::IdRef(heaps.variables[&(load.heap as u32)]),
                Operand::IdRef(load.index_id),
            ],
        ));
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(load.descriptor_type),
            Some(load.result_id),
            vec![Operand::IdRef(ptr_id)],
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

impl Heaps {
    fn new(module: &mut Module, ids: &mut IdAllocator) -> Self {
        let u32_type = find_or_insert_u32_type(module, ids);
        let untyped_ptr_type = ids.alloc();
        module.types_global_values.push(Instruction::new(
            Op::TypeUntypedPointerKHR,
            None,
            Some(untyped_ptr_type),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));

        Self {
            u32_type,
            untyped_ptr_type,
            variables: FxHashMap::default(),
            arrays: FxHashMap::default(),
            access_chain_ids: FxHashMap::default(),
        }
    }

    /// The `OpUntypedVariableKHR` for a heap, created on first use.
    fn variable_for(&mut self, module: &mut Module, ids: &mut IdAllocator, heap: DescriptorHeap) {
        if self.variables.contains_key(&(heap as u32)) {
            return;
        }

        let var = ids.alloc();
        module.types_global_values.push(Instruction::new(
            Op::UntypedVariableKHR,
            Some(self.untyped_ptr_type),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(var),
                Operand::Decoration(Decoration::BuiltIn),
                Operand::BuiltIn(heap.built_in()),
            ],
        ));
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(var),
                Operand::LiteralString(
                    match heap {
                        DescriptorHeap::Resource => "resource_heap",
                        DescriptorHeap::Sampler => "sampler_heap",
                    }
                    .into(),
                ),
            ],
        ));
        self.variables.insert(heap as u32, var);
    }

    /// The runtime array of a descriptor type, created on first use.
    ///
    /// Its stride is `OpConstantSizeOfEXT`, since only the implementation knows
    /// how large a descriptor is, and the constant has to be emitted *before* the
    /// array: `OpDecorateId`'s operand must appear earlier in the module than its
    /// target, which `spirv-val` enforces.
    fn array_for(&mut self, module: &mut Module, ids: &mut IdAllocator, descriptor_type: Word) {
        if self.arrays.contains_key(&descriptor_type) {
            return;
        }

        let stride = ids.alloc();
        module.types_global_values.push(Instruction::new(
            Op::ConstantSizeOfEXT,
            Some(self.u32_type),
            Some(stride),
            vec![Operand::IdRef(descriptor_type)],
        ));

        let array = ids.alloc();
        module.types_global_values.push(Instruction::new(
            Op::TypeRuntimeArray,
            None,
            Some(array),
            vec![Operand::IdRef(descriptor_type)],
        ));
        module.annotations.push(Instruction::new(
            Op::DecorateId,
            None,
            None,
            vec![
                Operand::IdRef(array),
                Operand::Decoration(Decoration::ArrayStrideIdEXT),
                Operand::IdRef(stride),
            ],
        ));
        self.arrays.insert(descriptor_type, array);
    }
}

/// `OpTypeInt 32 0`, which every module has, but not necessarily.
fn find_or_insert_u32_type(module: &mut Module, ids: &mut IdAllocator) -> Word {
    let existing = module.types_global_values.iter().find(|inst| {
        inst.class.opcode == Op::TypeInt
            && inst.operands[0].unwrap_literal_bit32() == 32
            && inst.operands[1].unwrap_literal_bit32() == 0
    });
    if let Some(inst) = existing {
        return inst.result_id.unwrap();
    }

    let id = ids.alloc();
    module.types_global_values.insert(
        0,
        Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ),
    );
    id
}

/// The value of a `u32` `OpConstant`, if `id` names one.
fn constant_u32(module: &Module, id: Word) -> Option<u32> {
    module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(id) && inst.class.opcode == Op::Constant)
        .map(|inst| inst.operands[0].unwrap_literal_bit32())
}

/// Declares the capability and extension, unless they are already there.
fn declare_capability(module: &mut Module) {
    let has_capability = module
        .capabilities
        .iter()
        .any(|inst| inst.operands[0].unwrap_capability() == Capability::DescriptorHeapEXT);
    if !has_capability {
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::DescriptorHeapEXT)],
        ));
    }

    // `DescriptorHeapEXT` implicitly declares `UntypedPointersKHR`, so only the
    // one extension is needed.
    let has_extension = module
        .extensions
        .iter()
        .any(|inst| inst.operands[0].unwrap_literal_string() == "SPV_EXT_descriptor_heap");
    if !has_extension {
        module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString("SPV_EXT_descriptor_heap".into())],
        ));
    }
}

/// Lists the heap variables in the interface of every entry point that reaches
/// one, transitively through calls.
fn add_heaps_to_entry_point_interfaces(module: &mut Module, heaps: &Heaps) {
    let heap_vars: FxHashSet<Word> = heaps.variables.values().copied().collect();

    // Which functions use a heap variable directly, and who calls whom.
    let mut uses_heaps: FxHashMap<Word, FxHashSet<Word>> = FxHashMap::default();
    let mut callees: FxHashMap<Word, Vec<Word>> = FxHashMap::default();
    for func in &module.functions {
        let func_id = func.def_id().unwrap();
        let (used, called) = (uses_heaps.entry(func_id).or_default(), &mut callees);
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::FunctionCall {
                    called
                        .entry(func_id)
                        .or_default()
                        .push(inst.operands[0].unwrap_id_ref());
                }
                for operand in &inst.operands {
                    if let Some(id) = operand.id_ref_any()
                        && heap_vars.contains(&id)
                    {
                        used.insert(id);
                    }
                }
            }
        }
    }

    for entry_point in &mut module.entry_points {
        let entry_fn = entry_point.operands[1].unwrap_id_ref();

        // Transitive closure over calls, so a heap used in a function that was
        // not inlined still reaches the entry point's interface.
        let mut needed = FxHashSet::default();
        let mut worklist = vec![entry_fn];
        let mut seen = FxHashSet::default();
        while let Some(func_id) = worklist.pop() {
            if !seen.insert(func_id) {
                continue;
            }
            needed.extend(uses_heaps.get(&func_id).into_iter().flatten().copied());
            worklist.extend(callees.get(&func_id).into_iter().flatten().copied());
        }

        let already: FxHashSet<Word> = entry_point.operands[3..]
            .iter()
            .filter_map(|operand| operand.id_ref_any())
            .collect();
        // Sorted, so the interface does not depend on hash iteration order.
        let mut to_add: Vec<Word> = needed.difference(&already).copied().collect();
        to_add.sort_unstable();
        entry_point
            .operands
            .extend(to_add.into_iter().map(Operand::IdRef));
    }
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
