Testing, Validation, Compatibility and Security Tests

Goals
- Ensure the integrity and performance of the "Megabinary Hash-Based Execution" model.
- Validate that the "fail-on-hash-mismatch" policy is infallible.
- Benchmark startup times specifically for agentic AI tool-use scenarios.

Test categories
1. **Hash-Based Execution Integrity**
   - **Mismatch Test**: Attempt to execute a modified binary or an unauthorized script. Verify that the vkernel denies execution.
   - **Path vs. Hash Test**: Move a recompiled binary to a different path and attempt execution. Verify that the hash-lookup still works (or fails correctly based on policy).
   - **Collision Test**: Fuzz the hash-lookup logic to ensure no edge cases allow unauthorized code to map to a valid megabinary entry point.

2. **Megabinary Functional Correctness**
   - **Dispatcher Test**: Verify that the global dispatcher correctly routes multiple different execution requests to their respective code segments within the megabinary.
   - **Library Sharing Test**: Ensure that shared libraries (libc, etc.) deduplicated within the megabinary behave correctly for all linked applications.

3. **Agentic AI Benchmarking**
   - **Tool Invocation Latency**: Measure the time from `execve` call to the first instruction of the recompiled tool (e.g., Python or a custom search tool). Target <10ms.
   - **Concurrency Test**: Run hundreds of concurrent agent tasks sharing the same AOT-compiled megabinary and measure memory pressure and vkernel overhead.

4. **Security & No-JIT Enforcement**
   - **JIT Bypass Attempt**: Run code within the megabinary that attempts to use `mmap` or `mprotect` to create executable memory. Verify that the vkernel blocks these calls.
   - **Boundary Fuzzing**: Fuzz the arguments passed between the vkernel and the WASM instances.

5. **Polyfill & Recompilation Coverage**
   - Run the recompiled versions of NGINX, Redis, and Python through their standard test suites to identify any lifted-code regressions.

Automation and CI
- Every megabinary build must pass a "Hash-Routing Verification" step before being signed.
- Use specialized runners with vkernel-enabled environments for automated integration tests.

Next steps
- Build the "Hash Mismatch" test suite.
- Develop the latency benchmarking tool for "instant" tool invocation.
