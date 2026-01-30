Testing, Validation, Compatibility and Security Tests

Goals
- Provide comprehensive testing strategies covering syscall correctness, compatibility, security, and performance.
- Find translation and vkernel bugs early through fuzzing and real-world application testing.

Test categories
1. Syscall conformance
   - Implement a subset of the Linux Test Project (LTP) focused on syscalls used by typical userland workloads.
   - Include tests for open/read/write, sockets, epoll, futex, signals, and clock/timers.

2. Functional compatibility
   - Run common applications (NGINX, Redis, Postgres client, Node/Express, Python/Flask) against behavior expectations.
   - Execute unit and integration test suites for these apps where available.

3. Polyfill validation
   - Test Node polyfill against popular NPM packages and a curated compatibility suite.
   - Test WASM CPython against common Python packages that do not require C extensions.

4. Security testing
   - Fuzz inputs to syscalls and the vkernel interface to discover edge cases.
   - Attempt to bypass no-JIT protections by invoking alternate syscall sequences.
   - Perform penetration testing and exploit analysis on the vkernel and loader components.

5. Performance benchmarking
   - Measure overhead for CPU-bound, I/O-bound, and network-bound workloads.
   - Measure latency and throughput for event-loop style apps (Node/QuickJS polyfill) vs. native.

Automation and CI
- Integrate tests into the pipeline to run after image recompile and before publishing.
- Provide a staging environment with vkernel enabled nodes for full-stack tests.

Observability and diagnostics
- Instrument vkernel and loader to produce structured logs for syscall usage and policy violations.
- Collect metrics on syscall counts, blocked syscalls, and performance hotspots.
- Produce detailed compatibility reports for images that fail tests, including suggested fixes.

Next steps
- Implement basic test harness and run a canonical "hello world" binary through the full pipeline and vkernel.
- Build fuzzing harness for vkernel syscall translator and loader.