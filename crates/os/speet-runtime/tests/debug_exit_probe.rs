use binary_io::{BinArch, BinOs};
use speet_recompile::frontend::{recompile_to_wasm_instrumented_plt, ExternalTargets};
use speet_recompile::plt::PltCallPlan;
use speet_runtime::{default_host_api, load_binary, HostApi, Runtime};

#[test]
fn debug_probe() {
    let mut rt = Runtime::new(std::sync::Arc::new(default_host_api()));
    if !rt.llvm_available() {
        eprintln!("no llvm, skip");
        return;
    }
    let guest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/c-corpus/aarch64-macos/exit.macho");
    let bin = load_binary(&guest).unwrap();
    eprintln!("entry={:#x}", bin.entry);
    let text = bin
        .sections
        .iter()
        .find(|s| s.name == "__text")
        .unwrap();
    let targets = ExternalTargets::from_imports(&bin.imports);
    let plt_plan = PltCallPlan::from_targets(&targets, rt.host.as_ref());
    eprintln!("plt_plan.by_addr = {:?}", plt_plan.by_addr);
    eprintln!("plt_plan.import_by_symbol = {:?}", plt_plan.import_by_symbol);
    let manifest = rt.host.import_manifest();
    let (wasm, unsupported) = recompile_to_wasm_instrumented_plt(
        &text.data,
        text.addr,
        bin.arch,
        Some(&plt_plan),
        Some(bin.entry),
        &manifest,
    );
    eprintln!("unsupported = {:?}", unsupported);
    std::fs::write("/tmp/exit_debug.wasm", &wasm).unwrap();
    let obj = rt.compile_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).unwrap();
    std::fs::write("/tmp/exit_debug.o", &obj).unwrap();
    let entry_param_count = speet_recompile::drive::entry_param_count(&wasm);
    let sp_idx = speet_recompile::drive::sp_param_index(BinArch::AArch64);
    let lr_idx = speet_recompile::drive::lr_param_index(BinArch::AArch64);
    let halt_addr = speet_recompile::frontend::halt_addr(text.addr, text.data.len());
    eprintln!("entry_param_count = {entry_param_count}, sp_param_index = {sp_idx}");
    let exe = std::path::Path::new("/tmp/exit_debug_exe");
    rt.link_guest_object(
        &obj, entry_param_count, sp_idx, halt_addr, lr_idx, BinArch::AArch64, BinOs::MacOs, exe,
    )
    .unwrap();
    eprintln!("linked to {}", exe.display());
}
