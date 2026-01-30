Threat Model and Security Properties

Goals
- Define attacker capabilities, security objectives, and mitigations.
- Provide clear criteria for acceptable residual risk.

Assumed attacker capabilities
- Attacker can supply or modify userland code inside a container image (e.g., malicious or compromised app).
- Attacker may obtain code execution inside the recompiled container process (userland), but not initial host privilege escalation.

Security objectives
- Prevent runtime code generation and execution of unsigned code.
- Limit attacker ability to use syscalls to escape or meaningfully compromise the host.
- Preserve integrity of attestation and audit logs.

Controls and mitigations
- Signed artifacts and loader verification ensure only pipeline-produced code executes.
- vkernel enforces syscall restrictions, prohibits executable memory allocation, and maps FDs and mounts.
- Host-level policies (seccomp-bpf, LSM) reduce kernel attack surface.
- Detailed audit logs and attestation allow detection and forensic analysis.

Residual risks
- Bugs in vkernel or loader could be exploited to bypass policies; mitigated via fuzzing, code review, and limited privilege for vkernel process.
- Polyfills may contain vulnerabilities; must be maintained and audited.
- Attackers could exploit permitted syscalls to perform expensive or DoS operations; mitigate with quotas and cgroup limits.

Response plan
- Terminate offending containers on policy violations and collect forensic data.
- Rotate keys and rebuild images if artifact signing keys are suspected to be compromised.

Next steps
- Perform threat modeling workshops centered on vkernel and loader attack surfaces and prioritize mitigations.