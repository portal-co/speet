30/60/90 Day Roadmap and Next Steps

30-day goals
- Prototype the "Megabinary Generator": a tool that merges multiple WASM modules into a single dispatch-capable blob.
- Define the `manifest.json` hash-mapping schema and the `execve` interception logic.
- Implement a minimal vkernel that can route execution requests from file hashes to WASM entry points.
- Recompile a simple set of tools (ls, cat, echo) into a single megabinary and execute them based on their original hashes.

60-day goals
- Build the "Hash-Based Execution Controller" into a containerd-compatible shim.
- Implement the "Global Container Linking" strategy: deduplicate common libraries within the megabinary.
- Add support for script shebang redirection: running `/bin/sh` scripts via a recompiled shell in the megabinary.
- Enforce the "fail-on-hash-mismatch" policy: ensure no non-recompiled code can execute.

90-day goals
- Optimize for **Agentic AI**: Integrate with a framework (e.g., LangChain or a custom tool-runner) to demonstrate <10ms tool invocation.
- Develop the "Pre-warmed Tooling" set: A high-performance megabinary containing Python, Git, and common CLI tools for agents.
- Expand vkernel syscall coverage to support NGINX and Redis within the megabinary model.
- Conduct a security audit focusing on the hash-verification and WASM-to-host boundary.

Longer term
- Scale to thousands of concurrent agent environments on a single host using shared AOT-compiled megabinaries.
- Automate the entire "Container to Megabinary" pipeline for arbitrary OCI images.
- Explore hardware-accelerated hash verification for even lower execution latency.

Immediate next steps (actionable)
- Finalize the Megabinary internal routing format (how to jump to the right code based on a hash).
- Update `14-formats.md` with the new hash-based manifest and megabinary layout.
- Prototype the hash-interception shim for `execve`.

Risks and mitigations
- **Megabinary Size**: Large containers might lead to huge WASM blobs. Mitigation: Use intelligent deduplication and only include truly necessary binaries.
- **Hash Collisions**: Use strong cryptographic hashes (e.g., BLAKE3) to eliminate collision risks.
- **Agent Orchestration**: Managing thousands of short-lived WASM instances requires efficient vkernel resource pooling.

Contact and collaboration
- Track the transition from "per-binary appending" to "single megabinary" through the updated project milestones.
