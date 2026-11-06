fn main() {
    println!("cargo:rerun-if-changed=resources/aeromesh.manifest");
    #[cfg(windows)]
    {
        let rc_available = std::process::Command::new("windres")
            .arg("--version")
            .output()
            .is_ok()
            || std::process::Command::new("rc.exe")
                .arg("/?")
                .output()
                .is_ok();
        if rc_available {
            let mut res = winres::WindowsResource::new();
            res.set_manifest_file("resources/aeromesh.manifest");
            let _ = res.compile();
        } else {
            println!("cargo:warning=Resource compiler (windres/rc.exe) not found; manifest not embedded into PE header");
        }
    }
}
