use super::*;

const DEFAULT_CTX_SIZE: u32 = 4096;

pub struct LlmPresetLoader {
    pub llm_preset: LlmPreset,
    pub available_vram: u32,
    pub use_ctx_size: Option<u32>,
}

impl Default for LlmPresetLoader {
    fn default() -> Self {
        Self {
            llm_preset: LlmPreset::Llama3_8bInstruct,
            available_vram: 12,
            use_ctx_size: None,
        }
    }
}

impl LlmPresetLoader {
    pub fn new() -> Self {
        LlmPresetLoader::default()
    }

    pub fn local_model_path(&self, hf_loader: &HuggingFaceLoader) -> Result<PathBuf> {
        // Presumes a batch_size of 512
        let q_bits = quantization_from_vram(
            self.llm_preset.number_of_parameters(),
            self.available_vram,
            estimate_context_size(
                &self.llm_preset.model_config_json()?,
                self.use_ctx_size.unwrap_or(DEFAULT_CTX_SIZE) as i64,
                512,
            ),
        );
        let local_model_filename = hf_loader.load_file(
            self.llm_preset
                .f_name_for_q_bits(q_bits)
                .expect("Invalid quantization bits"),
            self.llm_preset.gguf_repo_id(),
        )?;

        HuggingFaceLoader::canonicalize_local_path(local_model_filename)
    }
}
