# Recompilation Goal List

Candidate workloads for Speet-style AOT recompilation: buggy or hard-to-fix software, retro binaries, and stacks with operational overreach (telemetry, permissions, drivers). Names anchor to widely cited examples; this is a prioritization aid, not an exhaustive catalog.

---

## 1. Buggy / painful to fix / mission-critical

| Target | Why it’s a recompiler goal | Public signal |
|--------|----------------------------|----------------|
| **Bun** (`oven-sh/bun`) | Fast-moving native + JS runtime; hang/crash reports in the issue tracker | e.g. [Bun issues](https://github.com/oven-sh/bun/issues) — startup hangs (#26810), dynamic-import regressions (#22743), macOS arm64 crashes (#21126), heavy-thread crashes (#17206) |
| **Electron apps** (Slack, Discord, Microsoft Teams, VS Code) | Structural RAM/process cost (Chromium + Node per app) | Long-running threads (e.g. [VS Code #181806](https://github.com/microsoft/vscode/issues/181806)); general Electron resource critique in trade press |
| **SAP ERP / ABAP** | Decades of generated + custom code; upgrade/kernel interactions | KB/community-visible modes (PXA pressure, dynpro failures); [“FOR ALL ENTRIES disaster”](https://community.sap.com/t5/technology-blogs-by-sap/sap-support-case-quot-for-all-entries-disaster-quot/ba-p/12934163) |
| **Banking COBOL / batch** | Half-completed transfers, abends mid-batch, compiler/PTF cascades | Case-study narratives (e.g. [GlobalBank transfer incident writeup](https://datafield.dev/intermediate-cobol/part-06/chapter-32/case-study-01.html)); [2012 RBS outage](https://en.wikipedia.org/wiki/2012_RBS_Group_computer_system_problems) |
| **EHR access paths** | National-scale dependency even when the EHR binary is not the root fault | **July 2024 CrowdStrike** disruption to Epic/Cerner access ([Becker’s](https://www.beckershospitalreview.com/ehrs/global-it-outage-disrupts-epic-ehr-systems.html), [Modern Healthcare](https://www.modernhealthcare.com/cybersecruity/crowdstrike-outage-ehr-hospitals-epic-mychart)) |
| **Embedded OpenSSL (and peers)** | Firmware ships ancient TLS; updates are rare | **Heartbleed** CVE-2014-0160 ([CISA](https://www.cisa.gov/news-events/alerts/2014/04/08/openssl-heartbleed-vulnerability-cve-2014-0160)); embedded sunset risk ([Embedded Computing Design](https://www.embeddedcomputing.com/technology/security/software-security/openssl-sunset-heartbleed-in-embedded-apps)) |

---

## 2. Retro / preservation

| Target | ISA / OS | Notes |
|--------|----------|--------|
| **DOS / Win9x-era games** | x86 16/32-bit, Win32 | Examples often cited in preservation catalogs ([Abandonware DOS](https://www.abandonwaredos.com/) — e.g. *Thief: The Dark Project*, *Age of Empires II*, *Interstate ’76*) |
| **32-bit Windows utilities/games** | x86 | Still run under compatibility layers; good DLL/OS API coverage |
| **Classic Mac OS 9 / early OS X (PPC)** + legacy creative tools | **PowerPC** | Community path: QEMU / SheepShaver ([E-Maculation PPC QEMU](https://www.emaculation.com/doku.php/ppc-osx-on-qemu-for-osx)) |
| **Console preservation** (where legally obtained) | MIPS, ARM, PPC vary | Stress non-x86 guests when multiple ISAs are supported |

---

## 3. Overreach (network, permissions, drivers)

Recompilation does not replace policy or store review; these are targets where **sandboxing / syscall filtering / IPC rewiring** pairs naturally with translation.

| Target | Pattern | Anchor |
|--------|---------|--------|
| **TikTok** | Broad device/account data; SDK ecosystem | [WIRED](https://www.wired.com/story/tiktok-new-privacy-policy/), [Internet Safety Labs](https://internetsafetylabs.org/blog/research/tiktoks-real-privacy-risks/) |
| **Meta apps** (Facebook, Instagram, WhatsApp) | Large permission surface | Consumer investigations (e.g. [Which? on app permissions](https://www.which.co.uk/policy-and-insight/article/which-investigation-reveals-how-data-hungry-smartphone-apps-ask-for-shocking-levels-of-access-to-your-location-microphone-and-data-aZXTg4E5fiLt)) |
| **OEM printer/scanner suites** | Daemon + update + deep hooks; often slow to ship Apple Silicon builds | [Rosetta Check (Intel-only apps)](https://rosettacheck.com/blog/intel-apps-no-apple-silicon-2026), [Apple — Rosetta](https://support.apple.com/102527) |

---

## 4. Architecture matrix

| Architecture | Example targets |
|--------------|-----------------|
| **x86 / x86_64** | DOS/Win32 retro, Electron x64, Windows hospital clients, Bun on Linux/Windows, legacy Windows drivers |
| **A64 (ARM64)** | Bun on macOS ARM, Android ARM binaries, Apple Silicon native vs Intel-only splits ([Apple silicon / Rosetta](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment)) |
| **PowerPC** | Classic Mac PPC, some embedded/console lineages; emulation demand validates binary longevity ([E-Maculation](https://www.emaculation.com/doku.php/ppc-osx-on-qemu-for-osx)) |

---

## 5. Suggested sprint order

1. **Bun** — one modern binary, active bug corpus, good automation story.  
2. **One Electron app** (Discord, Slack, or VS Code) — memory, multi-process, Chromium surface.  
3. **One Win32 game** from preservation lists — x86 + Win32 + timing.  
4. **One Android ARM app** (permissions-heavy) — JNI + syscalls + sandbox (if in scope).  
5. **One PPC Mac binary** — non-Intel mac ABI stress.  
6. **One IoT/embedded image with bundled OpenSSL** — static linking, no source.

---

## Related

- [Container megabinary plan](container-plan.md) — how translated workloads may ship as megabinaries.
