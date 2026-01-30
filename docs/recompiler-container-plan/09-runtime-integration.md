Runtime Shim and Orchestration Integration

Goals
- Provide a runtime shim that launches recompiled containers, enforces host-level policies, and connects containers to the vkernel.
- Integrate with containerd and Kubernetes so recompiled images can be scheduled and managed with minimal disruption.

Runtime shim responsibilities
- Validate the image manifest and ensure the host has the required vkernel version and signer keys.
- Start the loader stub in a sandboxed process and bind it to the per-container vkernel session.
- Apply host-level seccomp-bpf, LSM, and cgroup policies before starting the process.
- Provide attestation and runtime metadata to orchestration systems (e.g., Kubernetes events/annotations).
- Stream auditing logs and metrics (syscalls attempted, blocked operations) to monitoring/ SIEM.

Integration points
1. Containerd / CRI
   - Implement a Containerd shim or CRI plugin that intercepts starts for recompiled images and starts the loader via the shim.
   - Ensure the shim sets correct capability drops and seccomp filters.

2. Kubernetes
   - Use a mutating admission controller or image policy webhook to ensure recompiled images are allowed only on nodes with vkernel.
   - Annotate Pods with vkernel attestation metadata and container verification status.

3. Node provisioning
   - Nodes that can run recompiled images must run vkernel service and have trust keys configured.
   - Provide node labels and taints to schedule recompiled workloads to appropriate nodes.

Operational workflows
- Deploying recompiled images: build and push recompiled image to registry; use admission webhook to ensure images are only scheduled on compatible nodes.
- Monitoring: collect vkernel logs, loader logs, and audit events; display in dashboards and forward to SIEM.
- Upgrades: allow rolling upgrade of vkernel, ensure backwards compatibility via manifest versioning.

Failure modes and mitigation
- If vkernel is unavailable: shim can refuse to start containers and surface clear error messages.
- If signature validation fails: deny start and provide detailed diagnostics to operators.
- If an application requires unsupported syscalls: shim should fail fast and surface compatibility report.

Developer and operator tooling
- CLI to run recompiled images locally with a local vkernel instance for dev testing.
- Debug mode to run loader with increased logging and to extract WASM blobs for inspection.

Next steps
- Prototype a simple shim that integrates with containerd's shimv2 API and starts loader stubs, passing a vkernel socket path and manifest info.