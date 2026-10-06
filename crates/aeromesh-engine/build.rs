use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_dir = root_dir.join("bin");

    if let Ok(lib_dir) = env::var("LLAMA_LIB_DIR") {
        let p = PathBuf::from(lib_dir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }

    let search_dirs = [
        bin_dir,
        root_dir.join("llama.cpp").join("build").join("src").join("Release"),
        root_dir.join("llama.cpp").join("build").join("bin").join("Release"),
        PathBuf::from(r"C:\AeroMesh_Latest\llama-cluster\llama.cpp\build\src\Release"),
        PathBuf::from(r"C:\AeroMesh_Latest\llama-cluster\llama.cpp\build\bin\Release"),
        PathBuf::from(r"C:\AeroMesh\AeroMesh-vap-2026\llama.cpp\build\src\Release"),
        PathBuf::from(r"C:\AeroMesh\AeroMesh-vap-2026\llama.cpp\build\bin\Release"),
    ];

    for dir in &search_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rerun-if-changed=build.rs");
}
