//! Thin runtime: on-demand native-to-native recompilation via tunneled HostApi.

mod cache;
mod link;
mod runtime;
mod toolchain;

pub use cache::{load_binary, load_text_from_object, ArtifactCache};
pub use link::link_guest;
pub use runtime::Runtime;
pub use toolchain::LlvmToolchain;

pub use speet_host_api::{
    default_host_api, FilteredHostApi, HostApi, HostApiRegistry, HostPolicy, TunneledHostApi,
};
