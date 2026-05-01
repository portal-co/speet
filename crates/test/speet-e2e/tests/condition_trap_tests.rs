//! Execution tests for the `ConditionTrap` mechanism.
//!
//! Three test scenarios, each building a small WASM module with an `if` block,
//! translating it through `WasmFrontend` with a condition trap installed, and
//! running the result in wasmi to verify the branch actually changed:
//!
//! 1. **FlipConditionTrap** — emits `i32.eqz`; branch is always inverted.
//! 2. **PassthroughHookTrap** — calls a `decide(i32) -> i32` import that returns
//!    its argument unchanged; the host verifies it was called.
//! 3. **RuntimeOverrideTrap** — same import, but the host always returns `0`;
//!    any nonzero input still takes the `else` branch (condition forced false).

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use speet_memory::AddressWidth;
use speet_traps::cond::{ConditionInfo, ConditionTrap};
use speet_traps::LocalDeclarator;
use speet_wasm::{GuestMemoryConfig, IndexOffsets, WasmFrontend};
use wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction, Module, TypeSection, ValType,
};
use wasmparser::BinaryReaderError;
use wasmi::{Engine, Linker, Module as WasmiModule, Store};

// ── Condition trap implementations ────────────────────────────────────────────

/// Emits `i32.eqz` to invert the condition.
struct FlipConditionTrap;

impl LocalDeclarator for FlipConditionTrap {}

impl ConditionTrap<(), BinaryReaderError> for FlipConditionTrap {
    fn on_condition(
        &self,
        _info: &ConditionInfo,
        _ctx: &mut (),
        go: &mut (dyn FnMut(&mut (), &Instruction<'_>) -> Result<(), BinaryReaderError> + '_),
    ) -> Result<(), BinaryReaderError> {
        go(&mut (), &Instruction::I32Eqz)
    }
}

/// Calls `call $decide` (import index 0).
///
/// The `decide(i32) -> i32` import receives the condition and returns a
/// (possibly different) i32 that becomes the new condition for the `if`.
/// This enables both transparent logging and runtime branch override.
struct HookConditionTrap {
    decide_fn_idx: u32,
}

impl LocalDeclarator for HookConditionTrap {}

impl ConditionTrap<(), BinaryReaderError> for HookConditionTrap {
    fn on_condition(
        &self,
        _info: &ConditionInfo,
        _ctx: &mut (),
        go: &mut (dyn FnMut(&mut (), &Instruction<'_>) -> Result<(), BinaryReaderError> + '_),
    ) -> Result<(), BinaryReaderError> {
        go(&mut (), &Instruction::Call(self.decide_fn_idx))
    }
}

// ── WASM module builders ──────────────────────────────────────────────────────

/// Build the input WASM module that WasmFrontend will translate.
///
/// ```wat
/// (module
///   (func (param i32) (result i32)
///     local.get 0
///     if (result i32)
///       i32.const 1
///     else
///       i32.const 0
///     end))
/// ```
fn build_input_wasm() -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);

    let mut codes = CodeSection::new();
    let mut f = Function::new([]);
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::If(BlockType::Result(ValType::I32)));
    f.instruction(&Instruction::I32Const(1));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    codes.function(&f);

    let mut module = Module::new();
    module.section(&types);
    module.section(&funcs);
    module.section(&codes);
    module.finish()
}

/// Translate `input` through WasmFrontend with the given condition trap, then
/// assemble a runnable WASM module with an optional `"env" "decide"` import.
fn translate_and_assemble(
    input: &[u8],
    trap: Option<Box<dyn ConditionTrap<(), BinaryReaderError>>>,
    include_decide_import: bool,
) -> Vec<u8> {
    let mut frontend: WasmFrontend<(), BinaryReaderError> =
        WasmFrontend::with_wasm_encoder_fn(
            vec![GuestMemoryConfig { addr_width: AddressWidth::W32, mapper: None }],
            0,
            IndexOffsets::default(),
        );
    if let Some(t) = trap {
        frontend.set_condition_trap(t);
    }

    let mut linker: speet_linker::Linker<'_, '_, (), BinaryReaderError, Function> =
        speet_linker::Linker::new();
    frontend.translate_module(&mut (), &mut linker, input).unwrap();

    // Take the translated function body.
    let (translated_fn, _func_type) = frontend.take_compiled().pop().expect("one function");

    // Assemble output module.
    let mut types = TypeSection::new();
    // type 0: (i32) -> i32  — used for both the test function and the decide import
    types.ty().function([ValType::I32], [ValType::I32]);

    let mut imports = ImportSection::new();
    let test_fn_idx;
    if include_decide_import {
        // import "env" "decide" at index 0; test function at index 1
        imports.import("env", "decide", wasm_encoder::EntityType::Function(0));
        test_fn_idx = 1u32;
    } else {
        test_fn_idx = 0u32;
    }

    let mut funcs = FunctionSection::new();
    funcs.function(0); // type 0

    let mut exports = ExportSection::new();
    exports.export("test", ExportKind::Func, test_fn_idx);

    let mut codes = CodeSection::new();
    codes.function(&translated_fn);

    let mut module = Module::new();
    module.section(&types);
    if include_decide_import {
        module.section(&imports);
    }
    module.section(&funcs);
    module.section(&exports);
    module.section(&codes);
    module.finish()
}

// ── wasmi execution helpers ───────────────────────────────────────────────────

fn call_test(wasm: &[u8], input: i32, decide_fn: Option<impl Fn(i32) -> i32 + Send + Sync + 'static>) -> i32 {
    let engine = Engine::default();

    type State = ();
    let mut store: Store<State> = Store::new(&engine, ());
    let mut linker: Linker<State> = Linker::new(&engine);

    if let Some(decide) = decide_fn {
        linker
            .func_wrap("env", "decide", move |v: i32| -> i32 { decide(v) })
            .unwrap();
    }

    let module = WasmiModule::new(&engine, wasm).expect("valid wasm");
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let func = instance.get_typed_func::<i32, i32>(&mut store, "test").unwrap();
    func.call(&mut store, input).unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn flip_trap_inverts_branch() {
    let input_wasm = build_input_wasm();

    // Without trap: if(1) → 1, if(0) → 0.
    let baseline = translate_and_assemble(&input_wasm, None, false);
    assert_eq!(call_test(&baseline, 1, Option::<fn(i32) -> i32>::None), 1);
    assert_eq!(call_test(&baseline, 0, Option::<fn(i32) -> i32>::None), 0);

    // With FlipConditionTrap: condition is negated before `if`.
    let flipped = translate_and_assemble(
        &input_wasm,
        Some(Box::new(FlipConditionTrap)),
        false,
    );
    // input=1 → condition=1 → i32.eqz → 0 → else branch → result 0
    assert_eq!(call_test(&flipped, 1, Option::<fn(i32) -> i32>::None), 0);
    // input=0 → condition=0 → i32.eqz → 1 → then branch → result 1
    assert_eq!(call_test(&flipped, 0, Option::<fn(i32) -> i32>::None), 1);
}

#[test]
fn logging_hook_records_condition_without_changing_it() {
    let input_wasm = build_input_wasm();

    // HookConditionTrap calls decide(i32) -> i32 at import index 0.
    // The passthrough decide host returns its argument unchanged → original branch behavior.
    let log1: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let log1_clone = Arc::clone(&log1);

    let wasm = translate_and_assemble(
        &input_wasm,
        Some(Box::new(HookConditionTrap { decide_fn_idx: 0 })),
        true,
    );

    // Test with input=1: decide called with 1, returns 1 → then branch → result 1.
    let result = call_test(&wasm, 1, Some(move |v: i32| -> i32 {
        log1_clone.lock().unwrap().push(v);
        v // passthrough
    }));
    assert_eq!(result, 1, "passthrough should not change result");
    let logged = log1.lock().unwrap().clone();
    assert_eq!(logged, vec![1], "decide saw condition=1");

    // Test with input=0.
    let log2: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let log2_clone = Arc::clone(&log2);
    let result2 = call_test(&wasm, 0, Some(move |v: i32| -> i32 {
        log2_clone.lock().unwrap().push(v);
        v
    }));
    assert_eq!(result2, 0, "passthrough should not change result for input=0");
    let logged2 = log2.lock().unwrap().clone();
    assert_eq!(logged2, vec![0], "decide saw condition=0");
}

#[test]
fn runtime_hook_overrides_condition_for_backtracking() {
    let input_wasm = build_input_wasm();

    let wasm = translate_and_assemble(
        &input_wasm,
        Some(Box::new(HookConditionTrap { decide_fn_idx: 0 })),
        true,
    );

    // Host always returns 0 → forces else branch regardless of input.
    let result_when_forced_false =
        call_test(&wasm, 1, Some(|_: i32| -> i32 { 0 }));
    assert_eq!(result_when_forced_false, 0, "forced-false: else branch returns 0");

    // Host always returns 1 → forces then branch regardless of input.
    let result_when_forced_true =
        call_test(&wasm, 0, Some(|_: i32| -> i32 { 1 }));
    assert_eq!(result_when_forced_true, 1, "forced-true: then branch returns 1");
}
