use tauri::{AppHandle, Emitter, Manager};

// ---------------------------------------------------------------------------
// Backend commands — the backend is a pure execution engine.
// It never decides when to start, stop, show, hide, or type.
// The frontend owns every decision.
// ---------------------------------------------------------------------------

/// Tell the sidecar to begin capturing audio.
/// The frontend calls this only after transitioning to Listening.
#[tauri::command]
pub fn begin_capture(app: AppHandle) -> Result<(), String> {
    // Emit to the main window so the STTApi can send "start_recording" to the sidecar
    let _ = app.emit("backend:begin_capture", ());
    Ok(())
}

/// Tell the sidecar to stop capturing audio.
/// The frontend calls this when the user releases the hotkey.
#[tauri::command]
pub fn end_capture(app: AppHandle) -> Result<(), String> {
    let _ = app.emit("backend:end_capture", ());
    Ok(())
}

/// Insert text into the focused window.
/// Uses Win32 clipboard + Ctrl+V (with SendInput Unicode fallback).
/// The frontend calls this after transcription completes.
#[tauri::command]
pub fn insert_text(text: String, restore_hwnd: Option<u64>) -> Result<bool, String> {
    if text.trim().is_empty() {
        return Ok(false);
    }
    let platform = std::env::consts::OS;
    match platform {
        "windows" => win32_insert(&text, restore_hwnd),
        "linux" => linux_insert(&text, restore_hwnd),
        "macos" => macos_insert(&text),
        _ => false,
    }
    .pipe(Ok)
}

/// Show the overlay window, positioned centered above the taskbar.
/// The frontend calls this when transitioning to Listening.
#[tauri::command]
pub fn show_overlay(app: AppHandle) -> Result<(), String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;

    // Position: centered horizontally, 80px above the bottom edge
    if let Ok(Some(monitor)) = win.primary_monitor() {
        let m_size = monitor.size();
        let m_pos = monitor.position();
        let pill_w = 280;
        let pill_h = 60;
        let margin_bottom = 80;
        let x = (m_pos.x + (m_size.width as i32 - pill_w) / 2) as f64;
        let y = (m_pos.y + m_size.height as i32 - pill_h - margin_bottom) as f64;
        let _ = win.set_position(tauri::Position::Physical(tauri::PhysicalPosition {
            x: x as i32,
            y: y as i32,
        }));
    }

    // Show without stealing focus
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::UI::WindowsAndMessaging::ShowWindow;
        use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNOACTIVATE;
        if let Some(hwnd) = win.hwnd().ok() {
            let _ = unsafe { ShowWindow(hwnd, SW_SHOWNOACTIVATE) };
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = win.show();
    }

    Ok(())
}

/// Hide the overlay window.
/// The frontend calls this when transitioning back to Ready.
#[tauri::command]
pub fn hide_overlay(app: AppHandle) -> Result<(), String> {
    let win = app
        .get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())?;
    let _ = win.hide();
    Ok(())
}

// ---------------------------------------------------------------------------
// Platform-specific text insertion
// ---------------------------------------------------------------------------

trait Pipe<T> {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(T) -> R;
}

impl<T> Pipe<T> for T {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(T) -> R,
    {
        f(self)
    }
}

#[cfg(target_os = "windows")]
fn win32_insert(text: &str, hwnd: Option<u64>) -> bool {
    use crate::win32;
    if let Some(h) = hwnd {
        win32::set_foreground_hwnd(h);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    if win32::set_clipboard(text) {
        std::thread::sleep(std::time::Duration::from_millis(30));
        win32::send_ctrl_v();
        return true;
    }
    win32::send_text_unicode(text);
    true
}

#[cfg(not(target_os = "windows"))]
fn win32_insert(_text: &str, _hwnd: Option<u64>) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn linux_insert(text: &str, hwnd: Option<u64>) -> bool {
    use crate::win32;
    if let Some(h) = hwnd {
        if h != 0 {
            win32::set_foreground_hwnd(h);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    if !win32::set_clipboard(text) {
        return false;
    }
    std::thread::sleep(std::time::Duration::from_millis(30));
    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    if is_wayland {
        let out = std::process::Command::new("wtype").arg(text).output();
        return out.map(|o| o.status.success()).unwrap_or(false);
    }
    win32::send_ctrl_v();
    true
}

#[cfg(not(target_os = "linux"))]
fn linux_insert(_text: &str, _hwnd: Option<u64>) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn macos_insert(text: &str) -> bool {
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!("tell application \"System Events\" to keystroke \"{escaped}\"");
    std::process::Command::new("osascript")
        .args(["-e", &script])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(not(target_os = "macos"))]
fn macos_insert(_text: &str) -> bool {
    false
}
