use crate::*;
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PinTracker {
    backing: Vec<bool>,
}
impl PinTracker {
    pub fn get(&self, a: usize) -> bool {
        return self.backing.get(a).cloned().unwrap_or_default();
    }
    pub fn flag(&mut self, a: usize) {
        while self.backing.len() <= a {
            self.backing.push(false);
        }
        self.backing[a] = true;
    }
}
