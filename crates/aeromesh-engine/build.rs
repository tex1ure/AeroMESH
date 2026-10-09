//! Link configuration for the native llama.cpp / GGML shared libraries.
//!
//! Resolution order for the import libraries (`llama`, `ggml`):
//!   1. `$LLAMA_LIB_DIR` (explicit override, highest priority)
//!   2. `<repo>/bin` (populated by `setup.ps1`)
//!   3. `<repo>/llama.cpp/build/{src,bin}/Release` (a local llama.cpp CMake build)
//!   4. `<repo>/target/{debug,release}` (DLLs copied next to the test/binary output)
//!
//! No absolute paths outside the repository are consulted: a checkout must be
//! buildable wherever the user cloned it.

use std::env;
use std::path::{Path, PathBuf};

/// Import-library file names produced by the MSVC and GNU toolchains.
const LINK_LIB_STEMS: [&str; 2] = ["llama", "ggml"];

fn import_lib_names(stem: &str) -> [String; 4] {
    [
        format!("{stem}.lib"),        // MSVC import library
        format!("{stem}.dll.a"),      // MinGW/GNU import library
        format!("lib{stem}.dll.a"),   // MinGW/GNU import library (prefixed)
        format!("lib{stem}.a"),       // static/archived form
    ]
}

fn dir_has_import_lib(dir: &Path, stem: &str) -> bool {
    import_lib_names(stem)
        .iter()
        .any(|name| dir.join(name).is_file())
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("aeromesh-engine must live at <root>/crates/aeromesh-engine")
        .to_path_buf();

    let mut search_dirs: Vec<PathBuf> = Vec::new();

    if let Ok(lib_dir) = env::var("LLAMA_LIB_DIR") {
        let p = PathBuf::from(&lib_dir);
        if p.is_dir() {
            search_dirs.push(p);
        } else {
            println!(
                "cargo:warning=LLAMA_LIB_DIR is set to '{lib_dir}', which is not an existing directory; ignoring it"
            );
        }
    }

    search_dirs.push(root_dir.join("bin"));
    search_dirs.push(root_dir.join("llama.cpp").join("build").join("src").join("Release"));
    search_dirs.push(root_dir.join("llama.cpp").join("build").join("bin").join("Release"));
    search_dirs.push(root_dir.join("target").join("release"));
    search_dirs.push(root_dir.join("target").join("debug"));

    let mut found_any = false;
    for dir in &search_dirs {
        if !dir.is_dir() {
            continue;
        }
        if LINK_LIB_STEMS.iter().any(|stem| dir_has_import_lib(dir, stem)) {
            println!("cargo:rustc-link-search=native={}", dir.display());
            found_any = true;
        }
    }

    if !found_any {
        println!(
            "cargo:warning=Could not locate llama/ggml import libraries (llama.lib, llama.dll.a, ...). \
             Run `pwsh ./setup.ps1` to populate `<repo>/bin`, build llama.cpp under `<repo>/llama.cpp/build`, \
             or set LLAMA_LIB_DIR to the directory holding the import libraries."
        );
    }

    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    for dir in &search_dirs {
        println!("cargo:rerun-if-changed={}", dir.display());
    }
}
