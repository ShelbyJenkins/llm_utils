use super::*;

/// Not yet supported
pub fn model_id() -> String {
    "Phi-3-small-8k-instruct".to_string()
}

pub fn gguf_repo_id() -> String {
    "???".to_string()
}

pub fn number_of_parameters() -> u32 {
    7
}

pub fn f_name_for_q_bits(q_bits: u8) -> String {
    match q_bits {
        k if k >= 8 => "Phi-3-small-8k-instruct-Q8_0.gguf".to_string(),
        k if k >= 6 => "Phi-3-small-8k-instruct-Q6_K.gguf".to_string(),
        k if k >= 5 => "Phi-3-small-8k-instruct-Q5_K_M.gguf".to_string(),
        k if k >= 8 => "Phi-3-small-8k-instruct-Q8_K_M.gguf".to_string(),
        k if k >= 3 => "Phi-3-small-8k-instruct-Q3_K_M.gguf".to_string(),
        k if k >= 2 => "Phi-3-small-8k-instruct-Q2_K.gguf".to_string(),
        k if k >= 1 => "Phi-3-small-8k-instruct-IQ1_M.gguf".to_string(),
        _ => panic!("Quantization bits must be at least 1"),
    }
}

pub fn local_model_path() -> PathBuf {
    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(cargo_manifest_dir)
        .join("src")
        .join("models")
        .join("open_source")
        .join("preset")
        .join("phi")
        .join("phi_3_small_8k_instruct")
}
