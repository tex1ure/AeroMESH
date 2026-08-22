use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_dir = root_dir.join("bin");
    let release_bin_dir = root_dir.join("llama.cpp").join("build").join("bin").join("Release");

    if bin_dir.exists() {
        println!("cargo:rustc-link-search=native={}", bin_dir.display());
    }
    if release_bin_dir.exists() {
        println!("cargo:rustc-link-search=native={}", release_bin_dir.display());
    }

    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rerun-if-changed=build.rs");
}
