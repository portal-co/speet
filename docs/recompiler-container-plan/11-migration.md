Incremental Rollout and Migration Plan

Goals
- Deploy the recompiler system gradually to minimize disruption and allow iterative improvements.
- Provide clear rollback and support pathways for app owners whose apps need changes.

Phased rollout
- Phase 0: Prototype
  - Recompile and run a single static C binary under local vkernel; validate basic I/O behaviors.

- Phase 1: Single-process containers
  - Support containers with a single recompiled process. Validate network and disk I/O.

- Phase 2: Multi-process and shared libs
  - Add futex, pthreads, signals support. Recompile containers with multiple processes.

- Phase 3: Polyfills for complex stacks
  - Introduce polyfill images for Node and Python; work with app teams to adapt.

- Phase 4: Full images and orchestration
  - Integrate with containerd and Kubernetes; deploy recompiled apps at low-volume canary routes.

- Phase 5: Fleet-wide rollout
  - After sufficient testing, expand scheduling of recompiled images to more nodes. Continually expand syscall coverage.

Developer migration steps
- Provide a compatibility report for each image recompiled, listing any replaced runtimes, missing syscalls, or required changes.
- Offer a dev CLI for local testing and debugging with a local vkernel.
- Provide polyfill images and guides for porting native modules to WASM.

Operational practices
- Only schedule recompiled images on nodes running compatible vkernel versions.
- Maintain image tagging policies (e.g., repo/image:recompiled-v1) to differentiate images.
- Monitor failures and maintain a rapid rollback mechanism by switching to the original image in case of major issues.

Support and escalation
- Create a support channel for app owners to report compatibility issues.
- Maintain runbooks for operator responses to policy violations and vkernel faults.

Next steps
- Start with a small set of internal services and iterate through phases, collecting feedback and compatibility data.