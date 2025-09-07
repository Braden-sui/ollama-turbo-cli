#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ptr::null_mut;

use serde::{Deserialize, Serialize};
use windows::core::{PCWSTR, PWSTR};
use windows::Win32::Foundation::FILETIME;
use windows::Win32::Security::Credentials::{CredDeleteW, CredFree, CredReadW, CredWriteW, CREDENTIALW, CRED_FLAGS, CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC};

const APP_PREFIX: &str = "Agent Desk Pro/";

fn to_wide(s: &str) -> Vec<u16> {
  use std::os::windows::ffi::OsStrExt;
  std::ffi::OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
}

#[tauri::command]
fn get_secret(key: String) -> Result<Option<String>, String> {
  let target = format!("{}{}", APP_PREFIX, key);
  let target_w = to_wide(&target);
  let mut pcred: *mut CREDENTIALW = null_mut();
  let res = unsafe { CredReadW(PCWSTR(target_w.as_ptr()), CRED_TYPE_GENERIC, 0, &mut pcred) };
  if res.is_ok() {
    unsafe {
      let cred = *pcred;
      let size = cred.CredentialBlobSize as usize;
      if size == 0 || cred.CredentialBlob.is_null() {
        CredFree(pcred as _);
        return Ok(None);
      }
      let slice = std::slice::from_raw_parts(cred.CredentialBlob as *const u8, size);
      let value = String::from_utf8_lossy(slice).to_string();
      CredFree(pcred as _);
      Ok(Some(value))
    }
  } else {
    Ok(None)
  }
}

#[tauri::command]
fn set_secret(key: String, value: String) -> Result<(), String> {
  let target = format!("{}{}", APP_PREFIX, key);
  let target_w = to_wide(&target);
  let blob = value.as_bytes().to_vec();
  let mut cred = CREDENTIALW {
    Flags: CRED_FLAGS(0),
    Type: CRED_TYPE_GENERIC,
    TargetName: PWSTR(target_w.as_ptr() as _),
    Comment: PWSTR::null(),
    LastWritten: FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 },
    CredentialBlobSize: blob.len() as u32,
    CredentialBlob: blob.as_ptr() as *mut u8,
    Persist: CRED_PERSIST_LOCAL_MACHINE,
    AttributeCount: 0,
    Attributes: null_mut(),
    TargetAlias: PWSTR::null(),
    UserName: PWSTR::null(),
  };
  // Safety: CredWriteW copies buffers before returning
  let res = unsafe { CredWriteW(&mut cred as *mut CREDENTIALW, 0) };
  if res.is_ok() {
    Ok(())
  } else {
    Err("Failed to write to Windows Credential Manager".into())
  }
}

#[tauri::command]
fn delete_secret(key: String) -> Result<(), String> {
  let target = format!("{}{}", APP_PREFIX, key);
  let target_w = to_wide(&target);
  let res = unsafe { CredDeleteW(PCWSTR(target_w.as_ptr()), CRED_TYPE_GENERIC, 0) };
  if res.is_ok() { Ok(()) } else { Err("Failed to delete credential".into()) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppVersion { version: String }

#[tauri::command]
fn version() -> AppVersion {
  AppVersion { version: env!("CARGO_PKG_VERSION").to_string() }
}

fn main() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![get_secret, set_secret, delete_secret, version])
    .run(tauri::generate_context!())
    .expect("error while running Agent Desk Pro");
}
