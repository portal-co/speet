Strict No-JIT Policy and Enforcement

Goals
- Ensure that no executable code may be generated at runtime inside a recompiled container unless it is produced and signed by the trusted build pipeline.
- Provide defense-in-depth across build-time, loader-time, vkernel-time, and host kernel controls.

Policy principles
- Default deny: disallow syscalls and operations that enable runtime code generation by default; explicitly allow when justified.
- Signed artifacts only: only execute code pages that correspond to signed artifacts appended to binaries or provided as signed AOT blobs.
- No executable allocations: deny mmap/mprotect/ memfd_create requests that would create or modify pages with PROT_EXEC.

Enforcement layers
1. Build-time
   - Artifacts built by the pipeline are the only allowed executable blobs. Sign everything.
   - Prohibit shipping JIT engines unless they are strictly configured to disable runtime codegen (rare exception, heavily audited).

2. Loader-time
   - Loader validates signatures and refuses to run unsigned code or dynamically compile code.
   - Verify that initial code pages map to signed regions.

3. vkernel enforcement
   - Enforce seccomp-like syscall restrictions, deny PROT_EXEC changes, and refuse memfd_create with exec.
   - Monitor for suspicious syscall sequences and terminate on policy violations.

4. Host-level policies
   - Apply seccomp-bpf or LSM policies at the container boundary; use cgroups to restrict resource usage.
   - Employ kernel mechanisms (e.g., SELinux, AppArmor, Landlock) to reduce attack surface.

5. Attestation and runtime checks
   - vkernel issues a signed attestation of the verified initial code hash and running modules.
   - Monitor runtime integrity: ensure no new executable pages are created during execution.

Handling allowed exceptions
- Some workloads may legitimately require runtime codegen (rare). For such cases:
  - Require explicit admin approval and strict per-container policy manifest documenting the need.
  - Review and audit the provided JIT engine and limit its capabilities (e.g., networkless, no host fd access).
  - Restrict its allowable syscalls and runtime memory regions.

Response to violations
- Immediate termination of offending container process.
- Emit detailed audit logs (attempted syscalls, PID, container id, binary hash) to security logs and optionally to SIEM.
- Optionally quarantine the container and trigger automated incident response workflows.

Testing the policy
- Write tests that attempt to create executable memory via mmap/mprotect/memfd_create and ensure vkernel blocks them.
- Fuzz attempts to bypass protections using alternate syscall sequences.
- Test legitimate workflows that need non-JIT behaviour to avoid regressions.

Developer guidance
- Encourage AOT compilation and precompilation of scripts where possible.
- Provide tooling to produce snapshots for languages that support it (Node snapshots, Python freeze tools).

Next steps
- Implement syscall monitoring hooks in the vkernel prototype.
- Implement attestation signatures and integrate with orchestration for reporting.