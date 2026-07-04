//! Named host API backends.

use crate::HostApi;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of named [`HostApi`] implementations.
pub struct HostApiRegistry {
    backends: HashMap<String, Arc<dyn HostApi>>,
}

impl Default for HostApiRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register("tunneled", Arc::new(crate::TunneledHostApi::for_host()));
        reg
    }
}

impl HostApiRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, api: Arc<dyn HostApi>) {
        self.backends.insert(name.to_string(), api);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn HostApi>> {
        self.backends.get(name).cloned()
    }

    pub fn default_backend(&self) -> Arc<dyn HostApi> {
        self.get("tunneled")
            .unwrap_or_else(|| Arc::new(crate::TunneledHostApi::for_host()))
    }
}
