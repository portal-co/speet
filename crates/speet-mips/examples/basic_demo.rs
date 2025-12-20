use speet_mips::MipsRecompiler;
use rabbitizer::{Instruction, InstrCategory};
use wasm_encoder::Function;

fn main() {
    println!("MIPS to WebAssembly Recompiler Demo");
    println!("==================================");
    
    // Create a recompiler instance
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Example 1: Simple addition
    println!("\n1. Translating: add $t0, $t1, $t2");
    let add_instruction = Instruction::new(0x012A4020, 0x1000, InstrCategory::CPU); // add $t0, $t1, $t2
    
    recompiler.translate_instruction(&mut ctx, &add_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 2: Immediate addition
    println!("\n2. Translating: addi $t0, $t1, 42");
    let addi_instruction = Instruction::new(0x212A002A, 0x1000, InstrCategory::CPU); // addi $t0, $t1, 42
    
    recompiler.translate_instruction(&mut ctx, &addi_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 3: Load word
    println!("\n3. Translating: lw $t0, 4($t1)");
    let lw_instruction = Instruction::new(0x8D2A0004, 0x1000, InstrCategory::CPU); // lw $t0, 4($t1)
    
    recompiler.translate_instruction(&mut ctx, &lw_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 4: Store word
    println!("\n4. Translating: sw $t0, 4($t1)");
    let sw_instruction = Instruction::new(0xAD2A0004, 0x1000, InstrCategory::CPU); // sw $t0, 4($t1)
    
    recompiler.translate_instruction(&mut ctx, &sw_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 5: Branch equal
    println!("\n5. Translating: beq $t0, $t1, target");
    let beq_instruction = Instruction::new(0x11290004, 0x1000, InstrCategory::CPU); // beq $t0, $t1, 4

    recompiler.translate_instruction(&mut ctx, &beq_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();



    // Example 6: Jump (using a simpler encoding)
    println!("\n6. Translating: j target");
    let j_instruction = Instruction::new(0x08000000, 0x1000, InstrCategory::CPU); // j 0x0 (simpler)
    
    recompiler.translate_instruction(&mut ctx, &j_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 7: System call with callback
    println!("\n7. Translating: syscall (with callback)");
    
    let mut syscall_count = 0;
    let mut syscall_callback = |syscall: &speet_mips::SyscallInfo, _ctx: &mut (), callback_ctx: &mut speet_mips::CallbackContext<_, _>| {
        syscall_count += 1;
        println!("   SYSCALL detected at PC: 0x{:x}, count: {}", syscall.pc, syscall_count);
        // Could emit custom WebAssembly code here if needed
        callback_ctx.emit(&wasm_encoder::Instruction::Nop).ok();
    };
    
    recompiler.set_syscall_callback(&mut syscall_callback);
    
    let syscall_instruction = Instruction::new(0x0000000C, 0x1000, InstrCategory::CPU); // syscall
    
    recompiler.translate_instruction(&mut ctx, &syscall_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 8: Logical operations
    println!("\n8. Translating: and $t0, $t1, $t2");
    let and_instruction = Instruction::new(0x012A4024, 0x1000, InstrCategory::CPU); // and $t0, $t1, $t2
    
    recompiler.translate_instruction(&mut ctx, &and_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    println!("\n9. Translating: or $t0, $t1, $t2");
    let or_instruction = Instruction::new(0x012A4025, 0x1000, InstrCategory::CPU); // or $t0, $t1, $t2
    
    recompiler.translate_instruction(&mut ctx, &or_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 10: Shift operations
    println!("\n10. Translating: sll $t0, $t1, 2");
    let sll_instruction = Instruction::new(0x000A4080, 0x1000, InstrCategory::CPU); // sll $t0, $t1, 2
    
    recompiler.translate_instruction(&mut ctx, &sll_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    // Example 11: Jump register (using NOP instead for demo)
    println!("\n11. Skipping JR instruction due to rabbitizer limitations");
    
    // JR example skipped: rabbitizer lacks a prebuilt JR instruction in this demo.
    // (No translation emitted for JR in this demo.)
    
    // Example 12: Jump and link register (using a simpler encoding)
    println!("\n12. Translating: jalr $ra, $t0");
    let jalr_instruction = Instruction::new(0x01800008, 0x1000, InstrCategory::CPU); // jalr $ra, $t0
    
    recompiler.translate_instruction(&mut ctx, &jalr_instruction, &mut |locals| {
        println!("   Function locals: {:?}", locals.collect::<Vec<_>>());
        Function::new(Vec::new())
    }).unwrap();
    
    println!("\nDemo completed successfully!");
    println!("Total SYSCALL callbacks invoked: {}", syscall_count);
}