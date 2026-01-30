Strict No-JIT Policy and Enforcement

Goals
- Ensure that no executable code may be generated at runtime inside a recompiled container.
- Provide defense-in-depth across build-time, vkernel-time, and host kernel controls.
- Enable high-security execution for AI-generated code by stripping all JIT capabilities.

Policy principles
- **AOT-Only Execution**: Only code recompiled into the signed WASM megabinary can execute.
- **No Runtime Code Generation**: Deny all syscalls that allow creating or modifying executable memory (PROT_EXEC).
- **Content-Addressable Verification**: Execution is mapped to cryptographic file hashes to ensure no un-vetted code is run.

Enforcement layers
1. Build-time
   - The recompiler pipeline produces a single AOT-compiled megabinary.
   - All code is statically verified and signed.

2. vkernel enforcement
   - **Hash-Gate**: The vkernel intercepts `execve` and only proceeds if the file hash matches the signed `manifest.json`.
   - **Memory Policy**: Deny `mmap`, `mprotect`, and `memfd_create` requests that involve `PROT_EXEC`.
   - **WASM Constraints**: The WASM engine is configured in AOT or interpreter mode, with no JIT compiler enabled.

3. Host-level policies
   - Apply seccomp-bpf to the vkernel and shim processes to ensure they cannot create executable memory on the host.
   - Use LSM (e.g., Landlock or AppArmor) to strictly isolate the container's environment.

4. Attestation and runtime checks
   - The vkernel provides a signed attestation of the megabinary hash and the manifest used for execution.
   - Any attempt to load code from outside the megabinary triggers an immediate audit event and process termination.

AI Agent Security
- AI agents often need to execute small scripts. In this model, even if an agent generates a script, it **cannot** execute it unless it matches a hash already in the manifest.
- This forces the agent to use only the pre-approved tools and interpreters (e.g., a recompiled Python interpreter) already inside the megabinary.

Response to violations
- Immediate termination of the offending process or the entire container.
- Structured audit logs containing the attempted syscall, the binary hash (if applicable), and the container context.

Testing the policy
- **Exploit Simulation**: Run a "malicious" agent task that tries to download and run an ELF or write and run a shell script. Verify that the vkernel blocks both.
- **Syscall Fuzzing**: Ensure the `PROT_EXEC` blocking logic cannot be bypassed by obscure syscall combinations.

Developer guidance
- All application code and dependencies must be processed by the recompiler pipeline.
- Scripts must be executed via recompiled/polyfilled interpreters already present in the megabinary.

Next steps
- Implement the "No-JIT" memory policy in the vkernel prototype.
- Develop the audit logging format for policy violations.
