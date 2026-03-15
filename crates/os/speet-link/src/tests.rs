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
}
