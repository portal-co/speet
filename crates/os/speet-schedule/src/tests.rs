//! Unit tests for speet-schedule.

#[cfg(test)]
mod unit_tests {
    use crate::FuncSchedule;

    // ── FuncSchedule layout (no Linker needed) ────────────────────────────────

    #[test]
    fn func_schedule_cross_binary_layout() {
        let mut schedule: FuncSchedule<(), core::convert::Infallible, wasm_encoder::Function> =
            FuncSchedule::new();

        let slot0 = schedule.push(3, |_, _| unreachable!());
        let slot1 = schedule.push(5, |_, _| unreachable!());

        // Layout is final after pushes.
        assert_eq!(schedule.layout().base(slot0), 0);
        assert_eq!(schedule.layout().base(slot1), 3);
        assert_eq!(schedule.layout().total(), 8);
    }
}
