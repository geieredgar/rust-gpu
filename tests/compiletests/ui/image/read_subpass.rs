// build-pass
// compile-flags: -C target-feature=+InputAttachment

// ignore-test: `read_subpass` cannot currently emit a module that passes
// `spirv-val`, whatever it is called with.
//
// Vulkan requires the coordinate of a `SubpassData` `OpImageRead` to be a
// *constant* — `OpConstantComposite` of (0,0), or `OpConstantNull`
// (VUID-StandaloneSpirv-SubpassData-04660). `Image::read_subpass` takes its
// coordinate by reference and `OpLoad`s it in `asm!`, so what reaches the read
// is an instruction rather than a constant:
//
//     %19 = OpCompositeConstruct %v2int %int_0 %int_0
//     %21 = OpImageRead %v4float %20 %19
//
// Passing `IVec2::ZERO` instead of `IVec2::new(0, 0)` does not help: the value
// still round-trips through a local, so it is still materialized rather than
// being a constant.
//
// Two ways out, neither of them small enough to do in passing:
//
//  * fold an `OpCompositeConstruct` whose operands are all constants into an
//    `OpConstantComposite`, which is the general fix and would help other
//    shaders, but rewrites disassembly that ~25 checked-in expectations pin;
//  * have `read_subpass` emit `OpConstantNull` for the coordinate, which is
//    always valid because (0,0) is the only value Vulkan permits — but it means
//    silently ignoring the argument, so it is an API decision rather than a bug
//    fix.
//
// Ignored rather than deleted so that whichever is chosen has a test waiting.

use spirv_std::spirv;
use spirv_std::{Image, arch};

#[spirv(fragment)]
pub fn main(
    #[spirv(descriptor_set = 0, binding = 0, input_attachment_index = 0)] image: &Image!(subpass, type=f32, sampled=false),
    output: &mut glam::Vec4,
) {
    let coords = image.read_subpass(glam::IVec2::new(0, 0));
    *output = coords;
}
