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

#[path = "harness/mod.rs"]
mod harness;

use std::sync::{Arc, Mutex};

use harness::{
    FlipConditionTrap, HookConditionTrap, WasmTranslateConfig,
    translate_wasm, translate_wasm_with_decide_import, run_wasm_with_decide,
    wasm_branches,
};

// ── Local execution helper ────────────────────────────────────────────────────

/// Call the exported `"test"` function in `wasm` with `input: i32` and an
/// optional `"env" "decide"` host function.  Returns the `i32` result.
fn call_test(wasm: &[u8], input: i32, decide_fn: Option<impl Fn(i32) -> i32 + Send + Sync + 'static>) -> i32 {
    use wasmi::{Engine, Linker, Module as WasmiModule, Store};
    let engine = Engine::default();
    type State = ();
    let mut store: Store<State> = Store::new(&engine, ());
    let mut linker: Linker<State> = Linker::new(&engine);
    if let Some(decide) = decide_fn {
        linker.func_wrap("env", "decide", move |v: i32| -> i32 { decide(v) }).unwrap();
    }
    let module = WasmiModule::new(&engine, wasm).expect("valid wasm");
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let func = instance.get_typed_func::<i32, i32>(&mut store, "test").unwrap();
    func.call(&mut store, input).unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn flip_trap_inverts_branch() {
    let input_wasm = wasm_branches();

    // Without trap: if(1) → 1, if(0) → 0.
    let baseline = translate_wasm(
        &input_wasm,
        WasmTranslateConfig::plain(),
        &[("test", 0)],
    );
    assert_eq!(call_test(&baseline, 1, Option::<fn(i32) -> i32>::None), 1);
    assert_eq!(call_test(&baseline, 0, Option::<fn(i32) -> i32>::None), 0);

    // With FlipConditionTrap: condition is negated before `if`.
    let flipped = translate_wasm(
        &input_wasm,
        WasmTranslateConfig { mapper: None, cond_trap: Some(Box::new(FlipConditionTrap)) },
        &[("test", 0)],
    );
    // input=1 → condition=1 → i32.eqz → 0 → else branch → result 0
    assert_eq!(call_test(&flipped, 1, Option::<fn(i32) -> i32>::None), 0);
    // input=0 → condition=0 → i32.eqz → 1 → then branch → result 1
    assert_eq!(call_test(&flipped, 0, Option::<fn(i32) -> i32>::None), 1);
}

#[test]
fn logging_hook_records_condition_without_changing_it() {
    let input_wasm = wasm_branches();

    let log1: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let log1_clone = Arc::clone(&log1);

    let wasm = translate_wasm_with_decide_import(&input_wasm, "test");

    // Test with input=1: decide called with 1, returns 1 → then branch → result 1.
    let result = call_test(&wasm, 1, Some(move |v: i32| -> i32 {
        log1_clone.lock().unwrap().push(v);
        v
    }));
    assert_eq!(result, 1, "passthrough should not change result");
    assert_eq!(log1.lock().unwrap().clone(), vec![1], "decide saw condition=1");

    // Test with input=0.
    let log2: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let log2_clone = Arc::clone(&log2);
    let result2 = call_test(&wasm, 0, Some(move |v: i32| -> i32 {
        log2_clone.lock().unwrap().push(v);
        v
    }));
    assert_eq!(result2, 0, "passthrough should not change result for input=0");
    assert_eq!(log2.lock().unwrap().clone(), vec![0], "decide saw condition=0");
}

#[test]
fn runtime_hook_overrides_condition_for_backtracking() {
    let input_wasm = wasm_branches();
    let wasm = translate_wasm_with_decide_import(&input_wasm, "test");

    // Host always returns 0 → forces else branch regardless of input.
    let result_when_forced_false = run_wasm_with_decide(&wasm, "test", 1, |_| 0);
    assert_eq!(result_when_forced_false, 0, "forced-false: else branch returns 0");

    // Host always returns 1 → forces then branch regardless of input.
    let result_when_forced_true = run_wasm_with_decide(&wasm, "test", 0, |_| 1);
    assert_eq!(result_when_forced_true, 1, "forced-true: then branch returns 1");
}
