//! [`ArchPluginRecompiler`] — drives an [`ArchPlugin`]'s command-stream
//! protocol (`ArchPlugin::step` / [`ArchOp`]) against a real
//! `dyn ReactorContext<Context, E, FnType = F>`, the same environment every
//! native arch recompiler (`speet-x86_64`, `speet-riscv`, …) borrows. See
//! `docs/plugin-api.md` §3.6 for why `ArchPlugin` is a command stream
//! rather than a single call, and `docs/guides/plugin-api.md` §4 for the
//! "do not" rules around `ArchOp`.
//!
//! Mirrors the shape every native recompiler already uses (verified against
//! `speet-x86_64`'s `direct.rs`):
//! - [`setup_traps`](ArchPluginRecompiler::setup_traps) mirrors a native
//!   recompiler's own `setup_traps`: append the plugin's declared
//!   register-file params, then the trap chain's params, record the mark.
//! - [`translate_bytes`](ArchPluginRecompiler::translate_bytes) mirrors
//!   e.g. `X86Recompiler::translate_bytes`: the actual decode loop, driven
//!   here by repeated `ArchPlugin::step` calls instead of a hardware
//!   decoder. Each `OpenFn` follows the exact `rewind → declare_trap_locals
//!   → alloc_cell → next_with` sequence `X86Recompiler::init_function` uses.
//! - [`Recompile`] is also implemented, with a *thin* `drain_unit` (just
//!   `rctx.drain_fns()` + packaging) — translation itself never happens
//!   there, exactly mirroring `X86Recompiler`'s own `Recompile` impl. This
//!   makes `ArchPluginRecompiler` usable from `FuncSchedule` for future
//!   multi-binary plugin support, even though the current `frontend.rs`
//!   activation point calls `translate_bytes` directly (mirroring how the
//!   native arches are called there today).
//!
//! ## `Jump`/`IndirectJump`: explicit bytes, not `ReactorContext::jmp`/`ji`
//!
//! Native recompilers resolve a direct jump via `ReactorContext::jmp`, which
//! defers materializing the actual `return_call` to yecta's predecessor-
//! graph machinery (and can fold a single-predecessor jump target away
//! entirely — see `docs/guides/yecta.md`'s function-merging/constant-
//! folding notes). This adapter deliberately does **not** use `jmp`/`ji`/
//! `ji_with_params`: it forwards every param with an explicit
//! `LocalGet`/`feed` sequence, then seals with an explicit, manually-encoded
//! `Instruction::ReturnCall`/`ReturnCallIndirect` via `seal_fn` — the exact
//! same proven mechanism `Seal` uses, just with a terminal instruction this
//! adapter constructs itself instead of one the plugin supplied. This is a
//! deliberate v1 simplification: it forgoes yecta's predecessor-graph
//! optimizations for plugin-backed jumps in exchange for a small, auditable
//! code path that doesn't depend on understanding that machinery's full
//! semantics. `target_idx = base_func_offset() + (target_pc - base_pc)` —
//! *does* include `base_func_offset`, unlike `jmp`'s own relative
//! `FuncIdx`, because a raw `Instruction::ReturnCall`/`ReturnCallIndirect`
//! needs an absolute WASM function index (verified against
//! `X86Recompiler::rip_to_func_idx`, which omits the offset specifically
//! because `jmp`'s `FuncIdx` is relative — a detail that does not apply
//! here). No bounds check against the translated range: a plugin must only
//! emit `Jump`/`IndirectJump` target PCs within the unit it was given.
//! Out-of-bounds dispatch (`oob_jump`) is explicitly out of `ArchOp`'s v1
//! scope (see its module doc) — a real plugin author needing it is what
//! would motivate adding it to v2.
//!
//! `IndirectJump` resolves its table via `rctx.pool().handler.indirect_jump`
//! (the same `IndirectJumpHandler` a `TablePlugin` adapts into) — called
//! directly rather than through `ji`/`ji_with_params`, for the same reason.
//!
//! ## `Seal`'s snippet vs. `ReactorContext::seal_fn`'s single instruction
//!
//! `ArchOp::Seal` carries a whole [`CodeSnippet`] (potentially several
//! instructions), but `ReactorContext::seal_fn` takes one terminal
//! `Instruction`. This adapter feeds every instruction but the last via
//! `feed`, then passes the last to `seal_fn` as the terminal — an empty
//! snippet falls back to `Instruction::Unreachable` so the function is
//! always validly terminated.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use speet_link_core::context::ReactorContext;
use speet_link_core::recompiler::Recompile;
use speet_link_core::unit::BinaryUnit;
use speet_plugin_api::arch::{ArchOp, ArchPlugin};
use speet_plugin_api::error::PluginError;
use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;

use crate::replay::replay_snippet;
use crate::CodeSnippet;

fn decode_error<E: From<PluginError>>(message: String) -> E {
    E::from(PluginError::decode_failure(message))
}

/// [`Recompile::BinaryArgs`] for [`ArchPluginRecompiler`]: `base_pc` and
/// `guest_bytes` are needed by the adapter itself (PC↔`FuncIdx` arithmetic,
/// servicing `ArchOp::RequestBytes`); `plugin_args` is forwarded verbatim to
/// `ArchPlugin::reset_for_next_binary` — the plugin defines its own encoding
/// for it, mirroring how `Recompile::BinaryArgs` is per-implementation for
/// every native recompiler too.
pub struct ArchPluginBinaryArgs {
    pub base_pc: u64,
    pub guest_bytes: Vec<u8>,
    pub plugin_args: Vec<u8>,
}

/// Drives an [`ArchPlugin`] against a real `ReactorContext`. See the module
/// docs for the exact sequencing this mirrors from native recompilers.
pub struct ArchPluginRecompiler<Context, E, F> {
    plugin: Arc<dyn ArchPlugin>,
    base_pc: u64,
    guest_bytes: Vec<u8>,
    _marker: PhantomData<fn(Context, E, F)>,
}

impl<Context, E, F> ArchPluginRecompiler<Context, E, F> {
    pub fn new(plugin: Arc<dyn ArchPlugin>) -> Self {
        Self {
            plugin,
            base_pc: 0,
            guest_bytes: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Mirrors a native recompiler's own `setup_traps`: append the plugin's
    /// declared register-file params to the layout, then let the installed
    /// trap chain append its own params, then record the params/locals
    /// mark. Call once, before any `translate_bytes`/`drain_unit` call.
    /// Returns the total parameter count (mirrors every native
    /// `setup_traps`'s return value).
    pub fn setup_traps(
        &mut self,
        rctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        _ctx: &mut Context,
    ) -> u32 {
        for ty in self.plugin.declare_params() {
            rctx.layout_mut().append(1, ValType::from(ty));
        }
        rctx.declare_trap_params(&mut ());
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        mark.total_locals
    }
}

impl<Context, E: From<PluginError>, F> ArchPluginRecompiler<Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    /// The decode loop: repeatedly calls `ArchPlugin::step`, driving
    /// `rctx`/`ctx` according to each returned [`ArchOp`], until `Done`.
    /// `base_pc`/`guest_bytes` set the PC↔`FuncIdx` origin and the buffer
    /// `ArchOp::RequestBytes` slices are served from (`guest_bytes[0]`
    /// corresponds to guest address `base_pc`).
    pub fn translate_bytes(
        &mut self,
        ctx: &mut Context,
        rctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        guest_bytes: &[u8],
        base_pc: u64,
        make_fn: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        self.base_pc = base_pc;
        self.guest_bytes = guest_bytes.to_vec();

        let mut feedback: Option<Vec<u8>> = None;
        let mut tail_idx: Option<usize> = None;
        loop {
            let op = self.plugin.step(feedback.as_deref()).map_err(E::from)?;
            feedback = None;
            match op {
                ArchOp::OpenFn { len } => {
                    let mark = rctx.locals_mark();
                    rctx.layout_mut().rewind(&mark);
                    rctx.declare_trap_locals(&mut ());
                    let _cell = rctx.alloc_cell();
                    let f = make_fn(&mut rctx.layout().iter_since(&mark));
                    tail_idx = Some(rctx.next_with(ctx, f, len)?);
                }
                ArchOp::Feed { snippet } => {
                    let idx = tail_idx.expect("ArchOp::Feed before ArchOp::OpenFn");
                    feed_snippet(rctx, ctx, idx, &snippet)?;
                }
                ArchOp::Jump { target_pc, .. } => {
                    let idx = tail_idx.take().expect("ArchOp::Jump before ArchOp::OpenFn");
                    let params = rctx.locals_mark().total_locals;
                    for p in 0..params {
                        rctx.feed(ctx, idx, &Instruction::LocalGet(p))?;
                    }
                    let target_idx =
                        rctx.base_func_offset() + target_pc.wrapping_sub(self.base_pc) as u32;
                    rctx.seal_fn(ctx, idx, &Instruction::ReturnCall(target_idx))?;
                }
                ArchOp::IndirectJump { snippet, .. } => {
                    let idx = tail_idx.take().expect("ArchOp::IndirectJump before ArchOp::OpenFn");
                    let params = rctx.locals_mark().total_locals;
                    for p in 0..params {
                        rctx.feed(ctx, idx, &Instruction::LocalGet(p))?;
                    }
                    feed_snippet(rctx, ctx, idx, &snippet)?;
                    let kind = {
                        let mut fed = speet_link_core::context::FedContext::new(rctx, idx);
                        rctx.pool().handler.indirect_jump(ctx, &mut fed)?
                    };
                    let table = match kind {
                        yecta::IndirectJumpKind::Table(t) => t,
                        yecta::IndirectJumpKind::Ref => yecta::TableIdx(0),
                    };
                    let yecta::TypeIdx(ty_idx) = rctx.pool().ty;
                    let yecta::TableIdx(table_idx) = table;
                    rctx.seal_fn(
                        ctx,
                        idx,
                        &Instruction::ReturnCallIndirect {
                            type_index: ty_idx,
                            table_index: table_idx,
                        },
                    )?;
                }
                ArchOp::Seal { snippet } => {
                    let idx = tail_idx.take().expect("ArchOp::Seal before ArchOp::OpenFn");
                    seal_snippet(rctx, ctx, idx, &snippet)?;
                }
                ArchOp::RequestBytes { guest_addr, max_len } => {
                    let start = guest_addr.wrapping_sub(self.base_pc) as usize;
                    let end = (start + max_len as usize).min(self.guest_bytes.len());
                    feedback = Some(if start < end {
                        self.guest_bytes[start..end].to_vec()
                    } else {
                        Vec::new()
                    });
                }
                ArchOp::Done => {
                    if let Some(idx) = tail_idx.take() {
                        rctx.seal_fn(ctx, idx, &Instruction::Unreachable)?;
                    }
                    return Ok(());
                }
            }
        }
    }
}

fn feed_snippet<Context, E: From<PluginError>, F: InstructionSink<Context, E>>(
    rctx: &(dyn ReactorContext<Context, E, FnType = F> + '_),
    ctx: &mut Context,
    tail_idx: usize,
    snippet: &CodeSnippet,
) -> Result<(), E> {
    let mut fed = speet_link_core::context::FedContext::new(rctx, tail_idx);
    replay_snippet(snippet, ctx, &mut fed, decode_error)
}

/// Feed every instruction in `snippet` but the last via `feed`, then pass
/// the last to `seal_fn` as the terminal — see module docs. An empty
/// snippet seals with a bare `Instruction::Unreachable`.
fn seal_snippet<Context, E: From<PluginError>, F: InstructionSink<Context, E>>(
    rctx: &(dyn ReactorContext<Context, E, FnType = F> + '_),
    ctx: &mut Context,
    tail_idx: usize,
    snippet: &CodeSnippet,
) -> Result<(), E> {
    let count = count_instructions(snippet).map_err(decode_error)?;
    if count == 0 {
        return rctx.seal_fn(ctx, tail_idx, &Instruction::Unreachable);
    }

    use wasm_encoder::reencode::{utils, Reencode};
    use wasmparser::{BinaryReader, OperatorsReader};

    struct IdentityReencode;
    impl Reencode for IdentityReencode {
        type Error = core::convert::Infallible;
    }

    let reader = OperatorsReader::new(BinaryReader::new(&snippet.wasm, 0));
    let mut reencoder = IdentityReencode;
    for (i, op) in reader.into_iter().enumerate() {
        let op = op.map_err(|e| decode_error(alloc::format!("{e}")))?;
        let instr = utils::instruction(&mut reencoder, op)
            .map_err(|e| decode_error(alloc::format!("{e:?}")))?;
        if i + 1 == count {
            return rctx.seal_fn(ctx, tail_idx, &instr);
        }
        rctx.feed(ctx, tail_idx, &instr)?;
    }
    unreachable!("count was computed from the same byte stream just iterated")
}

fn count_instructions(snippet: &CodeSnippet) -> Result<usize, String> {
    use wasmparser::{BinaryReader, OperatorsReader};
    let reader = OperatorsReader::new(BinaryReader::new(&snippet.wasm, 0));
    let mut n = 0usize;
    for op in reader {
        op.map_err(|e| alloc::format!("{e}"))?;
        n += 1;
    }
    Ok(n)
}

impl<Context, E: From<PluginError>, F: InstructionSink<Context, E>> Recompile<Context, E, F>
    for ArchPluginRecompiler<Context, E, F>
{
    type BinaryArgs = ArchPluginBinaryArgs;

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        args: Self::BinaryArgs,
    ) {
        self.base_pc = args.base_pc;
        self.guest_bytes = args.guest_bytes;
        self.plugin.reset_for_next_binary(&args.plugin_args);
    }

    fn count_fns(&self, bytes: &[u8]) -> u32 {
        self.plugin.count_fns(bytes)
    }

    /// Thin — translation already happened via `translate_bytes`, called
    /// separately before this (mirrors `X86Recompiler::drain_unit`, see
    /// module docs).
    fn drain_unit(
        &mut self,
        rctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F> {
        let base_func_offset = rctx.base_func_offset();
        let fns = rctx.drain_fns();
        BinaryUnit {
            fns,
            base_func_offset,
            entry_points,
            func_types: Vec::new(),
            data_segments: Vec::new(),
            data_init_fn: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_link_core::context::ReactorAdapter;
    use speet_plugin_api::snippet::PluginValType;
    use speet_plugin_api::PResult;
    use spin::Mutex;
    use wasm_encoder::Function;
    use yecta::{LocalLayout, Mark, Pool, Reactor, TableIdx, TypeIdx};

    #[derive(Debug)]
    #[allow(dead_code)] // message text is only read via the derived Debug, on panic
    struct TestErr(String);
    impl From<PluginError> for TestErr {
        fn from(e: PluginError) -> Self {
            TestErr(alloc::format!("{e}"))
        }
    }

    /// Decodes nothing real — just plays back a fixed `ArchOp` sequence,
    /// one register param, ending with a `Seal` whose snippet has more than
    /// one instruction (exercising the feed-all-but-last/seal-last split)
    /// followed by `Done`.
    struct ScriptedArch {
        step: Mutex<u32>,
    }
    impl ArchPlugin for ScriptedArch {
        fn reset_for_next_binary(&self, _args: &[u8]) {}
        fn count_fns(&self, _bytes: &[u8]) -> u32 {
            1
        }
        fn declare_params(&self) -> Vec<PluginValType> {
            alloc::vec![PluginValType::I64]
        }
        fn step(&self, _feedback: Option<&[u8]>) -> PResult<ArchOp> {
            let mut step = self.step.lock();
            let op = match *step {
                0 => ArchOp::OpenFn { len: 1 },
                1 => ArchOp::Feed {
                    snippet: CodeSnippet::from_instructions(&[
                        Instruction::LocalGet(0),
                        Instruction::I64Const(1),
                        Instruction::I64Add,
                        Instruction::LocalSet(0),
                    ]),
                },
                2 => ArchOp::Seal {
                    snippet: CodeSnippet::from_instructions(&[
                        Instruction::LocalGet(0),
                        Instruction::Drop,
                        Instruction::Unreachable,
                    ]),
                },
                _ => ArchOp::Done,
            };
            *step += 1;
            Ok(op)
        }
    }

    fn reactor_adapter<'a>(
        reactor: &'a mut Reactor<(), TestErr, Function>,
        pool: Pool<'a, (), TestErr>,
    ) -> ReactorAdapter<'a, (), TestErr, Function, yecta::LocalPool> {
        ReactorAdapter {
            reactor,
            layout: LocalLayout::empty(),
            locals_mark: Mark {
                slot_count: 0,
                total_locals: 0,
            },
            pool,
            escape_tag: None,
        }
    }

    #[test]
    fn drives_open_feed_seal_done() {
        let mut reactor = Reactor::<(), TestErr, Function>::default();
        static TABLE: TableIdx = TableIdx(0);
        let pool = Pool {
            handler: &TABLE,
            ty: TypeIdx(0),
        };
        let mut rctx = reactor_adapter(&mut reactor, pool);

        let mut recompiler: ArchPluginRecompiler<(), TestErr, Function> =
            ArchPluginRecompiler::new(Arc::new(ScriptedArch { step: Mutex::new(0) }));
        recompiler.setup_traps(&mut rctx, &mut ());

        recompiler
            .translate_bytes(&mut (), &mut rctx, &[], 0, &mut |locals| {
                Function::new(locals.collect::<Vec<_>>())
            })
            .unwrap();

        let fns = rctx.drain_fns();
        assert_eq!(fns.len(), 1);

        let mut expected = Function::new([]);
        expected.instruction(&Instruction::LocalGet(0));
        expected.instruction(&Instruction::I64Const(1));
        expected.instruction(&Instruction::I64Add);
        expected.instruction(&Instruction::LocalSet(0));
        expected.instruction(&Instruction::LocalGet(0));
        expected.instruction(&Instruction::Drop);
        expected.instruction(&Instruction::Unreachable);
        expected.instruction(&Instruction::End);
        assert_eq!(fns[0], expected);
    }

    /// Three functions: 0 and 1 both `Jump` to 2 — proves the
    /// `FuncIdx(target_pc.wrapping_sub(base_pc))` arithmetic resolves to the
    /// correct target. Two distinct predecessors (rather than one) is
    /// deliberate: yecta inlines a jump target with exactly one predecessor
    /// (see `docs/guides/yecta.md` — function merging/constant folding), so
    /// a single-predecessor version of this test would pass even with a
    /// wrong target index, by accident, as long as the merge still happens.
    /// With two predecessors a real `ReturnCall(2)` must be materialized in
    /// both.
    struct JumpingArch {
        step: Mutex<u32>,
    }
    impl ArchPlugin for JumpingArch {
        fn reset_for_next_binary(&self, _args: &[u8]) {}
        fn count_fns(&self, _bytes: &[u8]) -> u32 {
            3
        }
        fn declare_params(&self) -> Vec<PluginValType> {
            alloc::vec![PluginValType::I64]
        }
        fn step(&self, _feedback: Option<&[u8]>) -> PResult<ArchOp> {
            let mut step = self.step.lock();
            let op = match *step {
                0 => ArchOp::OpenFn { len: 0 },
                1 => ArchOp::Jump {
                    target_pc: 0x1002,
                    params: 1,
                },
                2 => ArchOp::OpenFn { len: 0 },
                3 => ArchOp::Jump {
                    target_pc: 0x1002,
                    params: 1,
                },
                4 => ArchOp::OpenFn { len: 0 },
                5 => ArchOp::Seal {
                    snippet: CodeSnippet::from_instructions(&[Instruction::Unreachable]),
                },
                _ => ArchOp::Done,
            };
            *step += 1;
            Ok(op)
        }
    }

    #[test]
    fn resolves_jump_target_pc_to_func_idx() {
        let mut reactor = Reactor::<(), TestErr, Function>::default();
        static TABLE: TableIdx = TableIdx(0);
        let pool = Pool {
            handler: &TABLE,
            ty: TypeIdx(0),
        };
        let mut rctx = reactor_adapter(&mut reactor, pool);

        let mut recompiler: ArchPluginRecompiler<(), TestErr, Function> =
            ArchPluginRecompiler::new(Arc::new(JumpingArch { step: Mutex::new(0) }));
        recompiler.setup_traps(&mut rctx, &mut ());

        recompiler
            .translate_bytes(&mut (), &mut rctx, &[0u8; 3], 0x1000, &mut |locals| {
                Function::new(locals.collect::<Vec<_>>())
            })
            .unwrap();

        let fns = rctx.drain_fns();
        assert_eq!(fns.len(), 3);

        let mut expected_caller = Function::new([]);
        expected_caller.instruction(&Instruction::LocalGet(0));
        expected_caller.instruction(&Instruction::ReturnCall(2));
        expected_caller.instruction(&Instruction::End);
        assert_eq!(
            fns[0], expected_caller,
            "Jump to 0x1002 (base 0x1000) must resolve to FuncIdx(2), forwarding 1 param"
        );
        assert_eq!(fns[1], expected_caller);

        let mut expected_target = Function::new([]);
        expected_target.instruction(&Instruction::Unreachable);
        expected_target.instruction(&Instruction::End);
        assert_eq!(fns[2], expected_target);
    }
}
