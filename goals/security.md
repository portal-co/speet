# Security

← [goals.md](../goals.md)

See [docs/container-plan.md](../docs/container-plan.md) §No-JIT Policy and §Threat Model for the security architecture.

---

## Hardening

- [ ] **No-JIT enforcement in vkernel** — block `mmap`/`mprotect` with `PROT_EXEC` at the vkernel level
- [ ] **Signed megabinary + manifest** — vkernel verifies signature before any execution; sign both `container.wasm` and `manifest.json` as a unit
- [ ] **App-level hardening** (`speet-traps`): deploy `RopDetectTrap` and `CfiReturnTrap` as standard hooks in all architecture recompilers (blocked on trap integration — see [active.md](active.md))
- [ ] **Antimalware hooks** — hook points for scanning deferred stores and loaded values

## Access control

- [ ] **Syscall whitelisting per binary** — stored in `manifest.json`; enforced by vkernel
- [ ] **Agentic AI safety** — agents can only invoke pre-approved tools already present in the signed megabinary; LLM-generated scripts have no valid hash entry and are denied

## Attestation

- [ ] vkernel provides signed attestation of megabinary hash + manifest before execution
- [ ] Host kernel: seccomp-bpf on vkernel + shim processes; LSM (Landlock/AppArmor) for isolation
