use super::*;

pub fn model_id() -> String {
    "Mixtral-8x7B-Instruct-v0.1".to_string()
}

pub fn gguf_repo_id() -> String {
    "MaziyarPanahi/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string()
}

pub fn number_of_parameters() -> u32 {
    56
}

pub fn f_name_for_q_bits(q_bits: u8) -> String {
    match q_bits {
        k if k >= 8 => "Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf".to_string(),
        k if k >= 6 => "Mixtral-8x7B-Instruct-v0.1.Q6_K.gguf".to_string(),
        k if k >= 5 => "Mixtral-8x7B-Instruct-v0.1.Q5_K_M.gguf".to_string(),
        k if k >= 4 => "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf".to_string(),
        k if k >= 3 => "Mixtral-8x7B-Instruct-v0.1.Q3_K_M.gguf".to_string(),
        k if k >= 2 => "Mixtral-8x7B-Instruct-v0.1.Q2_K.gguf".to_string(),
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
        .join("mistral")
        .join("mixtral_8x7b_instruct_v0_1")
}
