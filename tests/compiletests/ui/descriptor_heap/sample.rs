// Samples a texture out of the descriptor heaps, with no descriptor set or
// binding in sight — the whole point of `SPV_EXT_descriptor_heap`.
//
// The capability and extension are declared by the linker when a heap is used, so
// this test deliberately passes no `-C target-feature=+DescriptorHeapEXT`.

// build-pass

use spirv_std::descriptor_heap::{ResourceHeap, SamplerHeap};
use spirv_std::glam::{Vec2, Vec4};
use spirv_std::image::Image2d;
use spirv_std::{Sampler, spirv};

#[spirv(fragment)]
pub fn main(#[spirv(flat)] texture: u32, uv: Vec2, output: &mut Vec4) {
    let image: Image2d = unsafe { ResourceHeap::get(texture) };
    let sampler: Sampler = unsafe { SamplerHeap::get(0) };
    *output = image.sample(sampler, uv);
}
