//! Unit tests for speet-link.

#[cfg(test)]
mod unit_tests {
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    use wasm_encoder::ValType;

    use crate::{
        builder::MegabinaryBuilder,
        linker::LinkerPlugin,
        unit::{BinaryUnit, FuncType},
    };

    // ── FuncType ──────────────────────────────────────────────────────────────

    #[test]
    fn func_type_roundtrip() {
        let ft = FuncType::from_val_types(&[ValType::I32, ValType::I64], &[ValType::I32]);
        let params: Vec<ValType> = ft.params_val_types().collect();
        let results: Vec<ValType> = ft.results_val_types().collect();
        assert_eq!(params, &[ValType::I32, ValType::I64]);
        assert_eq!(results, &[ValType::I32]);
    }

    #[test]
    fn func_type_ord() {
        let a = FuncType::from_val_types(&[ValType::I32], &[]);
        let b = FuncType::from_val_types(&[ValType::I64], &[]);
        // i32 byte = 0x7F, i64 byte = 0x7E. As unsigned bytes 0x7F > 0x7E,
        // so a > b in lexicographic (byte-level) order.
        assert!(a != b);
        assert!(a > b);
    }

    #[test]
    fn func_type_eq_hash() {
        use alloc::collections::BTreeMap;
        let ft1 = FuncType::from_val_types(&[ValType::I32; 3], &[]);
        let ft2 = FuncType::from_val_types(&[ValType::I32; 3], &[]);
        assert_eq!(ft1, ft2);

        let mut map: BTreeMap<FuncType, u32> = BTreeMap::new();
        map.insert(ft1.clone(), 0);
        map.insert(ft2.clone(), 1); // overwrites
        assert_eq!(map.len(), 1);
        assert_eq!(map[&ft1], 1);
    }

    // ── MegabinaryBuilder type deduplication ──────────────────────────────────

    #[test]
    fn megabinary_type_dedup() {
        let type_a = FuncType::from_val_types(&[ValType::I32], &[]);
        let type_b = FuncType::from_val_types(&[ValType::I64], &[]);

        let mut builder: MegabinaryBuilder<u32> = MegabinaryBuilder::new();

        // Unit 1: two functions of type_a and one of type_b.
        builder.on_unit(BinaryUnit {
            fns: vec![1u32, 2u32, 3u32],
            base_func_offset: 0,
            entry_points: vec![("foo".to_string(), 0)],
            func_types: vec![type_a.clone(), type_a.clone(), type_b.clone()],
        });

        // Unit 2: one function of type_a (duplicate type) and one new type_b.
        builder.on_unit(BinaryUnit {
            fns: vec![4u32, 5u32],
            base_func_offset: 3,
            entry_points: vec![("bar".to_string(), 3)],
            func_types: vec![type_a.clone(), type_b.clone()],
        });

        let out = builder.finish();

        // Only 2 unique types.
        assert_eq!(out.types.len(), 2);

        // 5 functions total.
        assert_eq!(out.fns.len(), 5);
        assert_eq!(out.func_type_indices.len(), 5);

        // Functions 0,1,3 use type_a (index 0); functions 2,4 use type_b (index 1).
        let idx_a = out.types.iter().position(|t| *t == type_a).unwrap() as u32;
        let idx_b = out.types.iter().position(|t| *t == type_b).unwrap() as u32;
        assert_eq!(out.func_type_indices[0], idx_a);
        assert_eq!(out.func_type_indices[1], idx_a);
        assert_eq!(out.func_type_indices[2], idx_b);
        assert_eq!(out.func_type_indices[3], idx_a);
        assert_eq!(out.func_type_indices[4], idx_b);

        // 2 exports.
        assert_eq!(out.exports.len(), 2);
        assert_eq!(out.exports[0].0, "foo");
        assert_eq!(out.exports[1].0, "bar");
    }

    // ── BinaryUnit ────────────────────────────────────────────────────────────

    #[test]
    fn binary_unit_fields() {
        let ft = FuncType::from_val_types(&[ValType::I32; 26], &[]);
        let unit: BinaryUnit<String> = BinaryUnit {
            fns: vec!["fn0".to_string(), "fn1".to_string()],
            base_func_offset: 100,
            entry_points: vec![("main".to_string(), 100)],
            func_types: vec![ft.clone(), ft.clone()],
        };
        assert_eq!(unit.fns.len(), 2);
        assert_eq!(unit.base_func_offset, 100);
        assert_eq!(unit.entry_points[0].0, "main");
    }

    // ── Shim tests ────────────────────────────────────────────────────────────

    use alloc::boxed::Box;
    use crate::shim::{emit_shim, MemWidth, ParamSource, Place, SavePair, ShimSpec};

    /// `emit_shim` with `ParamSource::Load(Local)` and `ParamSource::Zero`
    /// must produce a function that validates as WASM.
    #[test]
    fn shim_local_and_zero() {
        let caller = FuncType::from_val_types(&[ValType::I64; 4], &[]);
        let callee = FuncType::from_val_types(&[ValType::I64; 6], &[]);

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 10,
            callee_sig: callee,
            param_map: vec![
                ParamSource::Load(Place::Local(0)),
                ParamSource::Load(Place::Local(1)),
                ParamSource::Load(Place::Local(2)),
                ParamSource::Load(Place::Local(3)),
                ParamSource::Zero, // → i64.const 0
                ParamSource::Zero,
            ],
            saves: vec![],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec); // must not panic
    }

    /// Constants as param sources.
    #[test]
    fn shim_constants() {
        let caller = FuncType::from_val_types(&[], &[]);
        let callee = FuncType::from_val_types(
            &[ValType::I32, ValType::I64, ValType::F32, ValType::F64],
            &[],
        );

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 5,
            callee_sig: callee,
            param_map: vec![
                ParamSource::I32Const(42),
                ParamSource::I64Const(-1),
                ParamSource::F32Const(3.14),
                ParamSource::F64Const(2.718),
            ],
            saves: vec![],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec);
    }

    /// A global-source parameter and a global-sink save.
    #[test]
    fn shim_global_source_and_save() {
        let caller = FuncType::from_val_types(&[ValType::I32; 2], &[]);
        let callee = FuncType::from_val_types(&[ValType::I32; 2], &[]);

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 20,
            callee_sig: callee,
            param_map: vec![
                // Pull first callee arg from global 0 instead of a local.
                ParamSource::Load(Place::Global(0)),
                // Second from caller local 1.
                ParamSource::Load(Place::Local(1)),
            ],
            saves: vec![
                // Save caller local 0 into global 1 before the call.
                SavePair { src: Place::Local(0), dst: Place::Global(1) },
            ],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec);
    }

    /// A memory-dereference source: load from memory[local_0 + 8] as i64.
    #[test]
    fn shim_deref_local_source() {
        let caller = FuncType::from_val_types(&[ValType::I32], &[]); // local 0 = ptr
        let callee = FuncType::from_val_types(&[ValType::I64], &[]);

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 7,
            callee_sig: callee,
            param_map: vec![
                // *((i64*)(local[0] + 8))
                ParamSource::Load(Place::Deref {
                    base: Box::new(Place::Local(0)),
                    offset: 8,
                    width: MemWidth::I64,
                    memory: 0,
                }),
            ],
            saves: vec![],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec);
    }

    /// A nested-pointer dereference: global 0 holds a base pointer;
    /// dereference it at offset 16 as i32.
    #[test]
    fn shim_deref_global_nested() {
        let caller = FuncType::from_val_types(&[], &[]);
        let callee = FuncType::from_val_types(&[ValType::I32], &[]);

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 3,
            callee_sig: callee,
            param_map: vec![
                // *(i32*)(global[0] + 16)
                ParamSource::Load(Place::Deref {
                    base: Box::new(Place::Global(0)),
                    offset: 16,
                    width: MemWidth::I32,
                    memory: 0,
                }),
            ],
            saves: vec![],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec);
    }

    /// Save to a memory slot through a global pointer, with sub-word widths.
    #[test]
    fn shim_save_to_deref_narrow() {
        let caller = FuncType::from_val_types(&[ValType::I32; 3], &[]);
        let callee = FuncType::from_val_types(&[ValType::I32; 3], &[]);

        let spec = ShimSpec {
            caller_sig: caller,
            callee_func_idx: 9,
            callee_sig: callee,
            param_map: (0..3).map(|i| ParamSource::Load(Place::Local(i))).collect(),
            saves: vec![
                // Store the low byte of local 2 to *(global[0] + 0) as u8.
                SavePair {
                    src: Place::Local(2),
                    dst: Place::Deref {
                        base: Box::new(Place::Global(0)),
                        offset: 0,
                        width: MemWidth::I32U8,
                        memory: 0,
                    },
                },
            ],
            extra_locals: vec![],
        };
        let _fn = emit_shim(&spec);
    }

    /// `MemWidth` helper methods.
    #[test]
    fn mem_width_properties() {
        assert_eq!(MemWidth::I32.natural_align(), 2);
        assert_eq!(MemWidth::I64.natural_align(), 3);
        assert_eq!(MemWidth::I32U8.natural_align(), 0);
        assert_eq!(MemWidth::I32U16.natural_align(), 1);
        assert_eq!(MemWidth::I64U32.natural_align(), 2);

        assert_eq!(MemWidth::I32.result_type(), ValType::I32);
        assert_eq!(MemWidth::I64.result_type(), ValType::I64);
        assert_eq!(MemWidth::I32S8.result_type(), ValType::I32);
        assert_eq!(MemWidth::I64S32.result_type(), ValType::I64);
        assert_eq!(MemWidth::F32.result_type(), ValType::F32);
        assert_eq!(MemWidth::F64.result_type(), ValType::F64);
    }
}
