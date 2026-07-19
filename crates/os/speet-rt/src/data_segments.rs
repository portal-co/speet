//! Generate the C source embedding passive WASM data-segment bytes and the
//! `memory.init` copy helper wasm-blitz's native backends call into.
//!
//! `wasm-blitz`'s `MemoryInit` lowering (see `blitz-x86-64`/`blitz-aarch64`'s
//! `naive.rs`) references two external symbols per module: one
//! `__wasm_data_seg_{index}` byte array per data segment, and a single shared
//! `__wasm_memory_init_copy` function. Neither is emitted by wasm-blitz's own
//! object output (mirroring `__wasm_mem_pages`/`__wasm_memory_grow`) — this
//! module produces the C translation unit that defines them, compiled and
//! linked alongside the guest object. When there are no data segments, it
//! instead defines a no-op `__speet_data_init` stub (see
//! [`generate_data_segments_c`]'s doc for why that's always needed).

/// One data segment's bytes, in WASM data-segment index order (segment `i`
/// becomes `__wasm_data_seg_{i}`).
pub struct DataSegmentBytes<'a> {
    pub bytes: &'a [u8],
}

/// Emit C source defining `__wasm_data_seg_N` byte arrays and the
/// `__wasm_memory_init_copy` helper. Argument order of the generated helper
/// matches both backends' calling convention for `Instruction::MemoryInit`:
/// destination offset into `__wasm_mem`, segment base pointer, source offset
/// within the segment, byte length.
///
/// When `segments` is empty, this instead defines a no-op `__speet_data_init`
/// stub: `speet_recompile`'s `finish_module` only emits a real
/// `__speet_data_init` export (compiled into `guest.o`) when the guest has
/// data segments, but `entry_bridge`'s `__speet_start` always calls it
/// unconditionally (see its module doc for why a weak/`if`-guarded call
/// doesn't work on Mach-O) — so exactly one definition must always be linked,
/// and this is where it comes from when `guest.o` doesn't provide one.
pub fn generate_data_segments_c(segments: &[DataSegmentBytes]) -> String {
    if segments.is_empty() {
        return String::from("void __speet_data_init(void) {}\n");
    }

    let mut out = String::from(
        "#include <stdint.h>\n#include <string.h>\n\nextern uint8_t *__wasm_mem;\n\n",
    );

    for (i, seg) in segments.iter().enumerate() {
        // Not `static`: wasm-blitz's compiled `guest.o` references
        // `__wasm_data_seg_{i}` as an external relocation from a completely
        // separate translation unit, so this needs external linkage — a
        // `static` array with no in-TU use is otherwise dead code from this
        // compiler's point of view and gets silently dropped.
        if seg.bytes.is_empty() {
            // A zero-size array isn't valid C; `memory.init` with this
            // segment always copies 0 bytes, so the placeholder byte is
            // never read.
            out.push_str(&format!("const uint8_t __wasm_data_seg_{i}[1] = {{0}};\n"));
            continue;
        }
        out.push_str(&format!("const uint8_t __wasm_data_seg_{i}[] = {{"));
        for (j, b) in seg.bytes.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str(&b.to_string());
        }
        out.push_str("};\n");
    }

    out.push_str(
        "\nvoid __wasm_memory_init_copy(uint32_t dest_off, const void *seg_base, uint32_t src_off, uint32_t len) {\n\
        \x20   memcpy(__wasm_mem + dest_off, (const uint8_t *)seg_base + src_off, len);\n\
        }\n",
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_one_array_per_segment_and_the_copy_helper() {
        let segs = [
            DataSegmentBytes { bytes: b"hi\n" },
            DataSegmentBytes { bytes: &[] },
        ];
        let src = generate_data_segments_c(&segs);
        assert!(src.contains("const uint8_t __wasm_data_seg_0[] = {104,105,10};"));
        assert!(!src.contains("static const uint8_t __wasm_data_seg_0"));
        assert!(src.contains("const uint8_t __wasm_data_seg_1[1] = {0};"));
        assert!(src.contains("void __wasm_memory_init_copy(uint32_t dest_off, const void *seg_base, uint32_t src_off, uint32_t len)"));
        assert!(src.contains("memcpy(__wasm_mem + dest_off, (const uint8_t *)seg_base + src_off, len);"));
    }

    #[test]
    fn empty_segment_list_defines_a_no_op_data_init_stub() {
        let src = generate_data_segments_c(&[]);
        assert!(src.contains("void __speet_data_init(void) {}"));
        assert!(!src.contains("__wasm_memory_init_copy"));
        assert!(!src.contains("__wasm_data_seg_"));
    }
}
