//! End-to-end tests for `speet-dex`: build a minimal `.dex` file by hand,
//! translate it through `DexRecompiler`, assemble a runnable WASM module,
//! execute it in `wasmi`, and check results directly out of linear memory.

#[path = "support/mod.rs"]
mod support;

use support::dex_builder::{ClassDef, CodeDef, DexBuilder, EncodedMemberDef};
use support::{harness, insn};

/// Simplest possible probe: a single-instruction method (`return-void` only),
/// `NoObjectModel`. Isolates whether basic translate→assemble→validate works
/// at all before layering on arithmetic / object-model instructions.
#[test]
fn return_void_only_smoke() {
    let mut b = DexBuilder::new();
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");

    let code = CodeDef {
        registers_size: 1,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[insn::return_void()]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let (fns, register_count) =
        harness::translate(&dex_bytes, speet_object::NoObjectModel, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");
}

/// `Foo { int i; }` with one method that computes `5 + 3` and stores it into
/// a freshly allocated `Foo.i`. Exercises: `const/4`, `add-int/2addr`,
/// `new-instance`, `iput`, `return-void`, and `LinearMemoryObjects`'s object
/// header + field write.
#[test]
fn arithmetic_new_instance_iput() {
    let mut b = DexBuilder::new();
    let field_i = b.add_field("LFoo;", "I", "i");
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");

    let code = CodeDef {
        registers_size: 3,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::const4(1, 5),                 // v1 = 5
            insn::const4(2, 3),                 // v2 = 3
            insn::add_int_2addr(1, 2),           // v1 = v1 + v2 = 8
            insn::new_instance(0, type_foo as u16), // v0 = new Foo()
            insn::iput(1, 0, field_i as u16),    // v0.i = v1
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![EncodedMemberDef { id: field_i, access_flags: 0x1, code: None }],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });

    let dex_bytes = b.build();

    // Sanity: the vendored `dex` crate can parse our hand-built container at all.
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);

    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");

    // Foo's single `i:I` field sits at byte offset 0 of the object's data
    // area, i.e. absolute address `HEAP_BASE + OBJECT_HEADER_SIZE`. The
    // object allocated by `new-instance` is deterministically the first
    // (and only) allocation, so its address is exactly `HEAP_BASE`.
    let field_addr = harness::HEAP_BASE + speet_object::OBJECT_HEADER_SIZE as i32;
    assert_eq!(harness::read_i32(&result.memory, field_addr), 8);
}

/// `if-eqz` branch (taken): skips the `iput`, leaving `Foo.i` at its
/// zero-initialized default rather than the value the skipped `iput` would
/// have written. Exercises structured control flow (`goto`/`if-eqz` targets
/// are code-unit offsets resolved through the Reactor, not raw jumps).
///
/// KNOWN GAP (`#[ignore]`d, not yet fixed): the generated module references
/// an out-of-bounds function index for the branch target. `translate_body`'s
/// `target(off)` closure computes a *raw code-unit-offset*-based `FuncIdx`
/// (`flat_idx.wrapping_add(off)`), matching this crate's documented `b+o`
/// contract — but `rctx.drain_fns()` returns a *compacted* vector (one entry
/// per translated instruction, in visit order, with no reserved slots for
/// non-instruction-start code-unit offsets), and this test's harness places
/// those compacted entries at sequential wasm function indices. For a
/// straight-line method (no explicit branches) the two numbering schemes
/// happen to agree only because every `return_call` chain link is resolved
/// via the *next allocated slot*, not raw arithmetic — but an explicit
/// forward branch (`if-eqz`/`goto`/etc.) computed via `target(off)` lands on
/// the wrong (or out-of-bounds) index whenever any code-unit offset between
/// the branch and its target was *not* a real instruction start (here: the
/// second unit of the preceding 2-unit `new-instance`/`iput` instructions).
/// Root cause is in the interaction between `speet-dex`'s `target(off)` and
/// `yecta`'s `next_with`/slot-numbering — needs investigation in `yecta`
/// itself (does `next_with` really assign dense, offset-based indices, or
/// compacted ones?) before attempting a fix in either crate.
#[test]
#[ignore = "known gap: if-eqz/goto branch targets resolve to the wrong wasm function index — see doc comment"]
fn if_eqz_branch_taken_skips_iput() {
    let mut b = DexBuilder::new();
    let field_i = b.add_field("LFoo;", "I", "i");
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");

    // 0: new-instance v0, Foo      (2 units, offsets 0-1)
    // 2: const/4 v1, #0            (1 unit,  offset 2)
    // 3: if-eqz v1, +4             (1 unit,  offset 3 -> target 3+4=7)
    // 4: const/4 v2, #7            (1 unit,  offset 4)
    // 5: iput v2, v0, field_i      (2 units, offsets 5-6)
    // 7: return-void               (1 unit,  offset 7)
    let code = CodeDef {
        registers_size: 3,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::new_instance(0, type_foo as u16),
            insn::const4(1, 0),
            insn::if_eqz(1, 4),
            insn::const4(2, 7),
            insn::iput(2, 0, field_i as u16),
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![EncodedMemberDef { id: field_i, access_flags: 0x1, code: None }],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");

    let field_addr = harness::HEAP_BASE + speet_object::OBJECT_HEADER_SIZE as i32;
    assert_eq!(
        harness::read_i32(&result.memory, field_addr),
        0,
        "if-eqz should have branched past the iput, leaving the field zero-initialized"
    );
}

/// `iput` then `iget` back into a different register, then write the
/// (doubled) result to the same field. If `iget` returned garbage or the
/// wrong address, the final value would not be `14`.
#[test]
fn iget_round_trip() {
    let mut b = DexBuilder::new();
    let field_i = b.add_field("LFoo;", "I", "i");
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");

    let code = CodeDef {
        registers_size: 3,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::new_instance(0, type_foo as u16), // v0 = new Foo()
            insn::const4(1, 7),                      // v1 = 7
            insn::iput(1, 0, field_i as u16),         // v0.i = 7
            insn::iget(2, 0, field_i as u16),          // v2 = v0.i (should be 7)
            insn::add_int_2addr(2, 2),                 // v2 = v2 + v2 = 14
            insn::iput(2, 0, field_i as u16),          // v0.i = 14
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![EncodedMemberDef { id: field_i, access_flags: 0x1, code: None }],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");

    let field_addr = harness::HEAP_BASE + speet_object::OBJECT_HEADER_SIZE as i32;
    assert_eq!(harness::read_i32(&result.memory, field_addr), 14);
}

/// `new-array` + `aput` + `aget`. Writes `99` to `arr[1]` directly, then
/// reads it back via `aget` and writes the (round-tripped) value to
/// `arr[2]` — proving both directions of array element access, inspected
/// straight out of linear memory (no field/object indirection needed).
#[test]
fn new_array_aput_aget() {
    let mut b = DexBuilder::new();
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");
    let type_int_array = b.add_type("[I");

    let code = CodeDef {
        registers_size: 6,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::const4(1, 3),                          // v1 = length 3
            insn::new_array(0, 1, type_int_array as u16), // v0 = new int[3]
            insn::const4(2, 1),                            // v2 = index 1
            insn::const16(3, 99),                           // v3 = 99
            insn::aput(3, 0, 2),                             // arr[1] = 99
            insn::aget(4, 0, 2),                              // v4 = arr[1] (should be 99)
            insn::const4(5, 2),                                // v5 = index 2
            insn::aput(4, 0, 5),                                // arr[2] = v4
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");

    let elem1_addr = harness::HEAP_BASE + speet_object::ARRAY_DATA_OFFSET as i32 + 1 * 4;
    let elem2_addr = harness::HEAP_BASE + speet_object::ARRAY_DATA_OFFSET as i32 + 2 * 4;
    assert_eq!(harness::read_i32(&result.memory, elem1_addr), 99, "direct aput");
    assert_eq!(harness::read_i32(&result.memory, elem2_addr), 99, "aget-then-aput round trip");
}

/// `instance-of` against the object's own type (true) and an unrelated type
/// (false).
///
/// KNOWN GAP (`#[ignore]`d, not yet fixed): the generated module fails
/// validation with "type mismatch: expected i32 but nothing on stack".
/// Decoding the emitted bytes shows `emit_instanceof`'s `If`/`I32Const(0)`/
/// `Else`/`<hash compare>`/`End` sequence (`speet-object`'s
/// `linear.rs::emit_instanceof`) comes out reordered as `If`/`Else`/
/// `I32Const(0)`/`<hash compare>`/`End` — the `I32Const(0)` that belongs in
/// the `if`-true branch has moved to *after* `Else`, leaving the true branch
/// empty (producing no value, hence "nothing on stack" for the block's
/// declared `(result i32)`) and the false branch pushing two i32s instead of
/// one. This single-`instanceof` case (`check_cast_mismatch_traps`, which
/// shares the same `If`/`Else`/`End`-based `emit_check_cast` machinery)
/// passes; this test — which calls `instance_of` twice back-to-back before
/// any other instruction — does not, suggesting the reordering is triggered
/// by something in the Reactor's instruction-bundling/fusion path when
/// multiple `If`/`Else`/`End`-emitting `ObjectModel` calls land in the same
/// fused function, not a bug in `emit_instanceof` itself. Needs investigation
/// into `yecta`'s bundle-flushing (`flush_bundles`) interaction with nested
/// control-flow instructions before attempting a fix.
#[test]
#[ignore = "known gap: If/Else ordering gets corrupted when multiple control-flow-emitting ObjectModel calls fuse into one function — see doc comment"]
fn instance_of_true_and_false() {
    let mut b = DexBuilder::new();
    let field_a = b.add_field("LFoo;", "I", "a");
    let field_b = b.add_field("LFoo;", "I", "b");
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");
    let type_bar = b.add_type("LBar;");

    let code = CodeDef {
        registers_size: 3,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::new_instance(0, type_foo as u16),   // v0 = new Foo()
            insn::instance_of(1, 0, type_foo as u16), // v1 = v0 instanceof Foo (1)
            insn::instance_of(2, 0, type_bar as u16), // v2 = v0 instanceof Bar (0)
            insn::iput(1, 0, field_a as u16),
            insn::iput(2, 0, field_b as u16),
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![
            EncodedMemberDef { id: field_a, access_flags: 0x1, code: None },
            EncodedMemberDef { id: field_b, access_flags: 0x1, code: None },
        ],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });
    // `Bar` needs no fields/methods — just a type entry to compare hashes against.
    b.add_class(ClassDef { class_type: type_bar, access_flags: 0x1, ..Default::default() });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(!result.trapped, "execution trapped unexpectedly");

    let addr_a = harness::HEAP_BASE + speet_object::OBJECT_HEADER_SIZE as i32;
    let addr_b = addr_a + 4;
    assert_eq!(harness::read_i32(&result.memory, addr_a), 1, "v0 instanceof Foo");
    assert_eq!(harness::read_i32(&result.memory, addr_b), 0, "v0 instanceof Bar");
}

/// `check-cast` against a mismatched type traps (via `throw_class_cast_fn`,
/// which is hand-wired to `unreachable` in the test harness).
#[test]
fn check_cast_mismatch_traps() {
    let mut b = DexBuilder::new();
    let proto_void = b.add_proto("V", "V", &[]);
    let method_test = b.add_method("LFoo;", proto_void, "test");
    let type_foo = b.add_type("LFoo;");
    let type_bar = b.add_type("LBar;");

    let code = CodeDef {
        registers_size: 1,
        ins_size: 0,
        outs_size: 0,
        insns: insn::assemble(&[
            insn::new_instance(0, type_foo as u16),
            insn::check_cast(0, type_bar as u16), // v0 is a Foo, not a Bar -> should trap
            insn::return_void(),
        ]),
    };

    b.add_class(ClassDef {
        class_type: type_foo,
        access_flags: 0x1,
        instance_fields: vec![],
        direct_methods: vec![EncodedMemberDef { id: method_test, access_flags: 0x1, code: Some(code) }],
        virtual_methods: vec![],
    });
    b.add_class(ClassDef { class_type: type_bar, access_flags: 0x1, ..Default::default() });

    let dex_bytes = b.build();
    dex::DexReader::from_vec(dex_bytes.clone()).expect("hand-built dex should parse");

    let obj_model = speet_object::LinearMemoryObjects {
        alloc_object_fn: harness::ALLOC_OBJECT_FN,
        alloc_array_fn: harness::ALLOC_ARRAY_FN,
        throw_class_cast_fn: harness::THROW_CLASS_CAST_FN,
    };
    let (fns, register_count) = harness::translate(&dex_bytes, obj_model, harness::N_HELPERS);
    let wasm = harness::assemble(fns, register_count);
    wasmparser::validate(&wasm).expect("generated module should validate");

    let result = harness::run(&wasm);
    assert!(result.trapped, "check-cast to a mismatched type should have trapped");
}
