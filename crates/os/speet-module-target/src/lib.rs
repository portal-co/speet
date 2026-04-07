#![no_std]
pub trait ModuleTarget<Ctx,Err>{

}
pub trait ModuleTargetDeclarator<Ctx,Err>{
    fn declare_module(&mut self, module: &mut (dyn ModuleTarget<Ctx,Err> + '_)) -> Result<(),Err>;
}