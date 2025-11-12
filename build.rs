use std::path::Path;
use time::{format_description, OffsetDateTime};
use winresource::WindowsResource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set compile-time environment variable
    let format = format_description::parse("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    // Compile Protocol Buffers for tonic
    println!("cargo:rerun-if-changed=proto");
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;

    // Embed icon for Windows executable
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let icon_path = "assets/card.ico";
        if !Path::new(icon_path).exists() {
            println!("cargo:warning=Icon file {} not found!", icon_path);
            return Err(format!("Icon file {} not found", icon_path).into());
        }
        println!("cargo:warning=Embedding icon from {}", icon_path);
        WindowsResource::new()
            .set_icon(icon_path)
            .set("FileDescription", "Vecno Miner")
            .set("ProductName", "Vecno Miner")
            .set("ProductVersion", env!("CARGO_PKG_VERSION"))
            .set("LegalCopyright", "Copyright © 2025")
            .compile()?;
    }

    // Re-run build if icon file changes
    println!("cargo:rerun-if-changed=assets/favicon.ico");

    Ok(())
}