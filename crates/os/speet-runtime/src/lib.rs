//! Thin runtime: on-demand native-to-native recompilation via tunneled HostApi.

mod cache;
mod execve_hook;
mod integrated;
mod link;
mod runtime;
mod suitability;
mod toolchain;

pub use cache::{load_binary, load_text_from_object, ArtifactCache};
pub use execve_hook::{default_socket_path, obtain_remote, ping, socket_path, ObtainResponse};
pub use integrated::{
    IntegratedNativeRuntime, NativeRuntime, ObtainError, SuitabilityReport,
};
pub use link::{link_guest, link_guest_integrated};
pub use runtime::{Runtime, validate_wasm_public};
pub use suitability::analyze_imports;
pub use toolchain::LlvmToolchain;

pub use speet_host_api::{
    default_host_api, integrated_host_api, FilteredHostApi, HostApi, HostApiRegistry, HostPolicy,
    ImportManifest, TunneledHostApi,
};
