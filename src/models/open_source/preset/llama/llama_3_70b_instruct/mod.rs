use super::*;

pub fn model_id() -> String {
    "Meta-Llama-3-70B-Instruct".to_string()
}

pub fn gguf_repo_id() -> String {
    "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF".to_string()
}

pub fn number_of_parameters() -> u32 {
    70
}

pub fn f_name_for_q_bits(q_bits: u8) -> String {
    match q_bits {
        k if k >= 5 => "Meta-Llama-3-70B-Instruct.Q5_K_M.gguf".to_string(),
        k if k >= 4 => "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf".to_string(),
        k if k >= 3 => "Meta-Llama-3-70B-Instruct.Q3_K_M.gguf".to_string(),
        k if k >= 2 => "Meta-Llama-3-70B-Instruct.Q2_K.gguf".to_string(),
        k if k >= 1 => "Meta-Llama-3-70B-Instruct.IQ1_M.gguf".to_string(),
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
        .join("llama")
        .join("llama_3_70b_instruct")
}
