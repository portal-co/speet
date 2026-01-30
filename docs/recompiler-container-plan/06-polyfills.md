Polyfills for Unrecompilable Binaries

Goals
- Provide functional replacements or compatible runtimes for binaries that cannot be recompiled straightforwardly (e.g., V8-based Node.js, native Python extensions, JIT-based VMs).
- Avoid runtime code generation in polyfills; prefer interpreter-based or precompiled snapshot approaches.
- Minimize changes required from application maintainers while keeping the runtime secure and auditable.

Classes of difficult workloads
- JIT-based VMs (V8, JVM)
- Interpreters that assume native extension loading (Python with C extensions)
- Complex build systems or apps that rely on runtime code generation

Approaches
1. Replace the engine with a non-JIT interpreter
   - Example: Replace Node/V8 with QuickJS-based Node polyfill that provides a large subset of Node APIs.
   - Pros: prevents runtime JIT; keeps API surface similar.
   - Cons: incomplete compatibility; may require reworking native modules.

2. Precompile snapshots at build time
   - Capture application-specific code as a precompiled snapshot and run it in a runtime that can execute snapshots without JIT.
   - Practical when app code is controlled by the builder (e.g., monorepos).

3. Build interpreters/runtimes to WASM
   - Compile CPython, Ruby, or other interpreters to WASM and run them via the vkernel/WASI runtime.
   - Pros: avoids native JIT and keeps runtime in the verified artifact.
   - Cons: complexity and potential incompatibility with CPython C extensions.

4. Recompile native modules to WASM
   - For Node native modules (node-gyp), require building modules to WASM/AOT in the pipeline.
   - Provide tooling and guides for module authors to produce WASM builds.

5. Provide a compatibility image
   - Offer curated base images with polyfills for Node, Python, and common stacks.
   - Encourage developers to base their images on these to minimize surprises.

Developer and migration strategy
- Provide migration guides and diagnostics to help developers port native modules or rely on supported APIs.
- Offer a compatibility testing harness so developers can detect polyfill incompatibilities early.

Operational considerations
- Maintain a curated set of polyfills with versioning and CVE tracking.
- Track feature gaps and prioritize polyfill improvements based on user demand and security impact.

Tradeoffs and limitations
- Polyfills are costly to maintain and may lag behind upstream features.
- Some applications may be infeasible to run without changes (heavy native extensions, custom JIT-based optimizations).

Next steps
- Prototype a QuickJS-based Node polyfill covering common APIs (fs, http, net, timers, streams, promise microtasks).
- Build WASM CPython baseline and evaluate C-extension strategies (e.g., provide a limited C-API shim or require recompilation to WASM).