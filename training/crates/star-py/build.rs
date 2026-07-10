//! Adds macOS extension linker flags only for explicit packaging builds.

fn main() {
    if std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
