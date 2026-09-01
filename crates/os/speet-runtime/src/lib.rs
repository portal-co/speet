//! Thin runtime: on-demand native-to-native recompilation via tunneled HostApi.

mod cache;
mod execve_hook;
mod integrated;
mod link;
pub mod report;
pub mod rtd_protocol;
mod runtime;
mod suitability;
mod toolchain;
mod transform_backend;

pub use cache::{load_binary, load_text_from_object, ArtifactCache};
pub use execve_hook::{default_socket_path, obtain_remote, ping, socket_path, ObtainResponse};
pub use integrated::{
    HotFrontend, IntegratedNativeRuntime, NativeRuntime, ObtainError, SuitabilityReport,
};
pub use link::{link_guest, link_guest_integrated};
pub use report::{GuestInfo, PluginHashes, RecompileReport};
pub use runtime::{validate_wasm_public, Runtime};
pub use suitability::{
    analyze_imports, analyze_imports_with_model, analyze_imports_with_model_and_wired,
};
pub use toolchain::LlvmToolchain;
pub use transform_backend::SharedIntegratedRuntime;

pub use speet_host_api::{
    default_host_api, integrated_host_api, FilteredHostApi, HostApi, HostApiRegistry, HostPolicy,
    ImportManifest, PltRedirect, RedirectingHostApi, TunneledHostApi,
};
