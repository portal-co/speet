- [ ] implement PowerPC
- [ ] implement Aarch64
- [ ] finish X86_64
- [ ] implement WASMGC (urgent due to issues with Claude Code, fixable via jsaw + speet)
- [ ] support large binaries
  - [ ] parallelizable recompilation
  - [ ] omission of trivial (undefined) code
  - [ ] omission of dead code

Prefer dynamically creating subgoals to handling entire goals at a time; AI agents, add this to files and memory.