//! Replay an artifact JSON: rebuild the FuzzCase, run both engines, and
//! print the compare verdict plus per-facet differences.
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    // Tolerant rebuild (same padding rules as tests/replay_corpus.rs) so
    // pre-M4 16-GPR artifacts replay too.
    let case = speet_diff_core::case_from_json(&v);
    let o = speet_diff_core::run_oracle(&case);
    let r = speet_diff_core::run_recompiled(&case);
    let verdict = speet_diff_core::compare_outcomes(&case, o.as_ref().ok(), r.as_ref());
    match verdict {
        speet_diff_core::Comparison::Match => println!("Match"),
        speet_diff_core::Comparison::Divergence(desc) => println!("DIVERGENCE: {desc}"),
        speet_diff_core::Comparison::Skip(reason) => println!("SKIP: {reason:?}"),
    }
    println!("oracle exit: {:?}", o.as_ref().map(|x| x.exit.clone()));
    println!("recomp exit: {:?}", r.as_ref().map(|x| x.exit.clone()).map_err(|e| format!("{e:?}")));
    if let (Ok(o), Ok(r)) = (o, r) {
        for (i, (a, b)) in o.regs.gprs.iter().zip(r.regs.gprs.iter()).enumerate() {
            if a != b {
                println!("  gpr{i}: oracle={a:#x} recompiled={b:#x}");
            }
        }
        if o.regs.sp != r.regs.sp {
            println!("  sp: oracle={:#x} recompiled={:#x}", o.regs.sp, r.regs.sp);
        }
        for (i, (a, b)) in o.regs.flags().iter().zip(r.regs.flags().iter()).enumerate() {
            if a != b {
                println!("  flag{i}: oracle={a} recompiled={b}");
            }
        }
        for (off, old, new) in speet_diff_core::case::MemoryDiff::compute(&o.data, &r.data).changes.iter().take(4) {
            println!("  data diff @{off}: {old:02x?} -> {new:02x?}");
        }
    }
}
