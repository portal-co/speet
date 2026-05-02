//! Unit tests for speet-linker.

#[cfg(test)]
mod unit_tests {
    use crate::Linker;
    use speet_link_core::unit::{BinaryUnit, FuncType};
    use speet_schedule::FuncSchedule;
    use wasm_encoder::{Function, Instruction, ValType};

    type SchedErr = core::convert::Infallible;

    /// Build a minimal `BinaryUnit<Function>` with `n` empty WASM functions.
    fn dummy_unit_fn(n: usize, base: u32) -> BinaryUnit<Function> {
        let ft = FuncType::from_val_types(&[], &[]);
        BinaryUnit {
            fns: (0..n)
                .map(|_| {
                    let mut fn_ = Function::new(alloc::vec![]);
                    fn_.instruction(&Instruction::End);
                    fn_
                })
                .collect(),
            base_func_offset: base,
            entry_points: alloc::vec![],
            func_types: (0..n).map(|_| ft.clone()).collect(),
            data_segments: alloc::vec![],
            data_init_fn: None,
        }
    }

    #[test]
    fn func_schedule_correct_count_succeeds() {
        let mut linker: Linker<(), SchedErr> = Linker::with_plugin(());
        let mut schedule: FuncSchedule<(), SchedErr, Function> = FuncSchedule::new();
        schedule.push(2, |_, _| dummy_unit_fn(2, 0));
        linker.execute_schedule(schedule, &mut ()); // must not panic
    }

    #[test]
    #[should_panic(expected = "declared 2 fns but emit produced 1")]
    fn func_schedule_wrong_count_panics() {
        let mut linker: Linker<(), SchedErr> = Linker::with_plugin(());
        let mut schedule: FuncSchedule<(), SchedErr, Function> = FuncSchedule::new();
        // Declare 2 but emit 1 — should panic.
        schedule.push(2, |_, _| dummy_unit_fn(1, 0));
        linker.execute_schedule(schedule, &mut ());
    }
}
