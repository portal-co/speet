Polyfills for Unrecompilable Binaries

Goals
- Provide functional replacements for JIT-based or complex runtimes (Node.js, Python) within the single WASM megabinary.
- Ensure polyfills are AOT-compiled and integrated into the hash-based execution model.
- Optimize polyfills for agentic AI tasks (e.g., Python scripts run by agents).

Megabinary Integration
- Polyfills are no longer separate files; they are compiled as core components of the WASM megabinary.
- Multiple binaries in the original container can point to the same internal polyfill entry point (e.g., several Python scripts all routed to the single WASM-based Python interpreter).

Supported Polyfill Types
1. **JIT-less Interpreters**
   - **Node.js**: Replace V8 with QuickJS or a JIT-less build of Node, compiled to WASM.
   - **Python**: Use a WASM-compiled CPython or MicroPython.
2. **Hash-Routed Interpreters**
   - When the vkernel intercepts an execution request for `/usr/bin/python3`, it routes it to the WASM-based Python entry point in the megabinary, passing the script path as an argument.

Agentic AI Optimizations
- **Pre-warmed Interpreters**: The megabinary can include a "warmed-up" interpreter state to further reduce startup time for agent scripts.
- **Shared Libraries**: Polyfills share common WASM-compiled libraries (like `zlib` or `openssl`) within the megabinary to save space.

Security Properties
- **No-JIT Persistence**: Polyfills are AOT-compiled into WASM, ensuring they never need to generate executable memory at runtime.
- **Controlled API Surface**: Polyfills interact with the vkernel through a strict, whitelisted set of WASI/vkernel syscalls.

Handling Native Extensions
- For Python or Node native extensions, the pipeline attempts to recompile them to WASM and link them into the megabinary.
- If a native extension cannot be recompiled, the associated polyfill will report an error at runtime, maintaining the "no-unauthorized-code" guarantee.

Next Steps
- Integrate a WASM-compiled CPython into the megabinary prototype.
- Develop a strategy for bundling common Python packages (pip) into the megabinary's internal filesystem/data sections.
