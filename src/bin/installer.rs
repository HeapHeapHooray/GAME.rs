use std::fs;
use std::path::PathBuf;
use std::process::Command;
use anyhow::{anyhow, Result};

fn main() -> Result<()> {
    println!("=== GAME.rs Installer ===");

    // 1. Resolve running installer directory
    let installer_path = std::env::current_exe()?;
    let installer_dir = installer_path.parent().ok_or_else(|| anyhow!("Could not resolve installer parent directory"))?;

    // 2. Determine local binary target based on OS
    let (exe_name, source_binary) = if cfg!(windows) {
        let options = [
            installer_dir.join("executables/game_rs.exe"),
            installer_dir.join("executables/game_rs_windows.exe"),
            PathBuf::from("executables/game_rs.exe"),
            PathBuf::from("executables/game_rs_windows.exe"),
            PathBuf::from("game_rs.exe"),
            PathBuf::from("target/release/game_rs.exe"),
        ];
        let found = options.into_iter().find(|p| p.exists())
            .ok_or_else(|| anyhow!("Pre-built binary (game_rs.exe or game_rs_windows.exe) not found in standard locations."))?;
        ("game_rs.exe", found)
    } else if cfg!(target_os = "macos") {
        let options = [
            installer_dir.join("executables/game_rs_mac"),
            installer_dir.join("executables/game_rs"),
            PathBuf::from("executables/game_rs_mac"),
            PathBuf::from("executables/game_rs"),
            PathBuf::from("game_rs"),
            PathBuf::from("target/release/game_rs"),
        ];
        let found = options.into_iter().find(|p| p.exists())
            .ok_or_else(|| anyhow!("Pre-built binary (game_rs_mac or game_rs) not found in standard locations."))?;
        ("game_rs", found)
    } else {
        // Assume Linux
        let options = [
            installer_dir.join("executables/game_rs_linux"),
            installer_dir.join("executables/game_rs"),
            PathBuf::from("executables/game_rs_linux"),
            PathBuf::from("executables/game_rs"),
            PathBuf::from("game_rs"),
            PathBuf::from("target/release/game_rs"),
        ];
        let found = options.into_iter().find(|p| p.exists())
            .ok_or_else(|| anyhow!("Pre-built binary (game_rs_linux or game_rs) not found in standard locations."))?;
        ("game_rs", found)
    };
    println!("Found binary: {:?}", source_binary);

    // 2. Determine installation paths based on OS
    let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    let home_dir = std::env::var(home_var)
        .map_err(|_| anyhow!("Could not resolve user home directory (environment variable {} not found)", home_var))?;
    let home_path = PathBuf::from(home_dir);

    let (install_dir, bin_link_path) = if cfg!(windows) {
        let dir = home_path.join(".game_rs");
        (dir, None)
    } else {
        let dir = home_path.join(".local/share/game_rs");
        let link = home_path.join(".local/bin").join(exe_name);
        (dir, Some(link))
    };

    println!("Installing to directory: {:?}", install_dir);
    fs::create_dir_all(&install_dir)?;

    // 3. Copy binary to target directory
    let target_binary = install_dir.join(exe_name);
    println!("Copying binary to {:?}", target_binary);
    fs::copy(&source_binary, &target_binary)?;

    // 4. Download checkpoints from GitHub Releases
    let zip_temp_path = install_dir.join("checkpoints.zip");
    let checkpoint_url = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-large-onnx.zip";
    
    println!("Downloading checkpoints (approx. 50 MB) from: {}", checkpoint_url);
    let mut response = ureq::get(checkpoint_url).call()?;
    
    let mut zip_file = fs::File::create(&zip_temp_path)?;
    std::io::copy(&mut response.body_mut().as_reader(), &mut zip_file)?;
    println!("Download complete! Extracting model checkpoints...");

    // 5. Extract checkpoints ZIP recursively
    let zip_file = fs::File::open(&zip_temp_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)?;
    let target_checkpoints = install_dir.join("checkpoints");

    if target_checkpoints.exists() {
        println!("Removing existing target checkpoints directory...");
        fs::remove_dir_all(&target_checkpoints)?;
    }
    fs::create_dir_all(&target_checkpoints)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let raw_name = file.name();

        // Anonymize/normalize root directory name (GAME-1.0.3-large-onnx -> GAME-1.0-large-onnx)
        let renamed_path = if raw_name.starts_with("GAME-1.0.3-large-onnx/") {
            raw_name.replacen("GAME-1.0.3-large-onnx/", "GAME-1.0-large-onnx/", 1)
        } else if raw_name == "GAME-1.0.3-large-onnx" {
            "GAME-1.0-large-onnx".to_string()
        } else {
            raw_name.to_string()
        };

        let outpath = target_checkpoints.join(renamed_path);
        if file.is_dir() {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                fs::create_dir_all(p)?;
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    println!("Checkpoints successfully extracted!");

    // 6. Delete temporary downloaded ZIP file
    println!("Cleaning up temporary archive...");
    fs::remove_file(&zip_temp_path)?;

    // 7. Setup Path / Symlinks
    if let Some(link_path) = bin_link_path {
        // macOS/Linux Symlink setup
        let link_parent = link_path.parent().ok_or_else(|| anyhow!("Invalid symlink parent path"))?;
        fs::create_dir_all(link_parent)?;

        if link_path.exists() {
            println!("Removing existing symlink at {:?}", link_path);
            fs::remove_file(&link_path)?;
        }

        println!("Creating symlink at {:?}", link_path);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target_binary, &link_path)?;

        println!("\nInstallation completed successfully!");
        println!("Please make sure {:?} is in your PATH.", link_parent);
        println!("You can run the application with: {}", exe_name);
    } else {
        // Windows Path registry setup
        println!("Configuring user PATH environment variable...");
        let install_dir_str = install_dir.to_string_lossy().replace("/", "\\");
        let ps_command = format!(
            "$oldPath = [Environment]::GetEnvironmentVariable('Path', 'User'); \
             if (-not $oldPath.Split(';').Contains('{}')) {{ \
                 $newPath = \"$oldPath;{}\"; \
                 [Environment]::SetEnvironmentVariable('Path', $newPath, 'User'); \
                 write-host 'Successfully added to user PATH.'; \
             }} else {{ \
                 write-host 'Directory is already in user PATH.'; \
             }}",
            install_dir_str, install_dir_str
        );

        let output = Command::new("powershell")
            .args(["-Command", &ps_command])
            .output()?;

        if !output.status.success() {
            eprintln!("Warning: Failed to update PATH environment variable automatically.");
            eprintln!("Please add {:?} to your User PATH manually.", install_dir);
        } else {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("{}", stdout.trim());
        }

        println!("\nInstallation completed successfully!");
        println!("You may need to restart your terminal for the PATH changes to take effect.");
        println!("You can run the application with: {}", exe_name);
    }

    Ok(())
}
