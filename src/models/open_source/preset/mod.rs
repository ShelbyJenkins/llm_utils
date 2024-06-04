pub mod llama;
pub mod mistral;
pub mod phi;
use super::*;
pub use llama::*;
pub use mistral::*;
pub use phi::*;

pub struct PresetModelBuilder {
    pub hf_token: Option<String>,
    pub open_source_model_type: LlmPreset,
    pub quantization_from_vram_gb: u32,
    pub use_ctx_size: u32,
}

impl Default for PresetModelBuilder {
    fn default() -> Self {
        Self {
            hf_token: None,
            open_source_model_type: LlmPreset::default(),
            quantization_from_vram_gb: 12,
            use_ctx_size: 2048,
        }
    }
}

impl PresetModelBuilder {
    pub fn new() -> Self {
        PresetModelBuilder::default()
    }

    pub fn hf_token(&mut self, hf_token: &str) -> &mut Self {
        self.hf_token = Some(hf_token.to_string());
        self
    }

    pub fn quantization_from_vram(&mut self, quantization_from_vram_gb: u32) -> &mut Self {
        self.quantization_from_vram_gb = quantization_from_vram_gb;
        self
    }

    pub fn use_ctx_size(&mut self, use_ctx_size: u32) -> &mut Self {
        self.use_ctx_size = use_ctx_size;
        self
    }

    pub fn llama_3_8b_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Llama3_8bInstruct;
        self
    }

    pub fn llama_3_70b_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Llama3_70bInstruct;
        self
    }

    pub fn mistral_7b_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Mistral7bInstructV0_3;
        self
    }

    pub fn mixtral_8x7b_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Mixtral8x7bInstructV0_1;
        self
    }

    pub fn phi_3_medium_4k_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Phi3Medium4kInstruct;
        self
    }

    pub fn phi_3_mini_4k_instruct(&mut self) -> &mut Self {
        self.open_source_model_type = LlmPreset::Phi3Mini4kInstruct;
        self
    }

    pub async fn load(&mut self) -> Result<OsLlm> {
        let model_config_json = self.open_source_model_type.model_config_json()?;

        // Presumes a batch_size of 512
        let q_bits = quantization_from_vram(
            self.open_source_model_type.number_of_parameters(),
            self.quantization_from_vram_gb,
            safe_tensors::estimate_context_size(&model_config_json, self.use_ctx_size.into(), 512),
        );

        let local_model_filename = HuggingFaceLoader::new(self.hf_token.clone())
            .model_from_repo_id(&self.open_source_model_type.gguf_repo_id())
            .load_file(&self.open_source_model_type.f_name_for_q_bits(q_bits))
            .await?;

        let model_url = HuggingFaceLoader::model_url_from_repo_and_local_filename(
            &self.open_source_model_type.gguf_repo_id(),
            &local_model_filename.to_string_lossy(),
        );

        let local_model_path = HuggingFaceLoader::canonicalize_local_path(local_model_filename)?;
        Ok(OsLlm {
            model_id: self.open_source_model_type.model_id(),
            model_url,
            model_config_json,
            chat_template: self.open_source_model_type.chat_template()?,
            local_model_paths: vec![local_model_path],
            tokenizer: Some(self.open_source_model_type.tokenizer()),
        })
    }
}

#[derive(Clone, Default, Debug)]
pub enum LlmPreset {
    #[default]
    Llama3_8bInstruct,
    Llama3_70bInstruct,
    Mistral7bInstructV0_3,
    Mixtral8x7bInstructV0_1,
    Phi3Medium4kInstruct,
    Phi3Mini4kInstruct,
}

impl LlmPreset {
    pub fn model_id(&self) -> String {
        match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::model_id(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::model_id(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::model_id(),
            LlmPreset::Mixtral8x7bInstructV0_1 => mixtral_8x7b_instruct_v0_1::model_id(),
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::model_id(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::model_id(),
        }
    }

    pub fn gguf_repo_id(&self) -> String {
        match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::gguf_repo_id(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::gguf_repo_id(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::gguf_repo_id(),
            LlmPreset::Mixtral8x7bInstructV0_1 => mixtral_8x7b_instruct_v0_1::gguf_repo_id(),
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::gguf_repo_id(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::gguf_repo_id(),
        }
    }

    pub fn model_config_json(&self) -> Result<OsLlmConfigJson> {
        let local_model_path = match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::local_model_path(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::local_model_path(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::local_model_path(),
            LlmPreset::Mixtral8x7bInstructV0_1 => mixtral_8x7b_instruct_v0_1::local_model_path(),
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::local_model_path(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::local_model_path(),
        };
        let config_json_path = local_model_path.join("config.json");
        model_config_json_from_local(&config_json_path)
    }

    pub fn chat_template(&self) -> Result<OsLlmChatTemplate> {
        let local_model_path = match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::local_model_path(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::local_model_path(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::local_model_path(),
            LlmPreset::Mixtral8x7bInstructV0_1 => mixtral_8x7b_instruct_v0_1::local_model_path(),
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::local_model_path(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::local_model_path(),
        };
        let tokenizer_config_json_path = local_model_path.join("tokenizer_config.json");
        chat_template_from_local(&tokenizer_config_json_path)
    }

    pub fn tokenizer(&self) -> LlmTokenizer {
        let local_model_path = match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::local_model_path(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::local_model_path(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::local_model_path(),
            LlmPreset::Mixtral8x7bInstructV0_1 => mixtral_8x7b_instruct_v0_1::local_model_path(),
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::local_model_path(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::local_model_path(),
        };
        let tokenizer_json_path = local_model_path.join("tokenizer.json");
        LlmTokenizer::new_from_tokenizer_json(&tokenizer_json_path)
    }

    pub fn f_name_for_q_bits(&self, q_bits: u8) -> String {
        match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::f_name_for_q_bits(q_bits),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::f_name_for_q_bits(q_bits),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::f_name_for_q_bits(q_bits),
            LlmPreset::Mixtral8x7bInstructV0_1 => {
                mixtral_8x7b_instruct_v0_1::f_name_for_q_bits(q_bits)
            }
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::f_name_for_q_bits(q_bits),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::f_name_for_q_bits(q_bits),
        }
    }

    pub fn number_of_parameters(&self) -> f64 {
        let base_parameters = match self {
            LlmPreset::Llama3_8bInstruct => llama_3_8b_instruct::number_of_parameters(),
            LlmPreset::Llama3_70bInstruct => llama_3_70b_instruct::number_of_parameters(),
            LlmPreset::Mistral7bInstructV0_3 => mistral_7b_instruct_v0_3::number_of_parameters(),
            LlmPreset::Mixtral8x7bInstructV0_1 => {
                mixtral_8x7b_instruct_v0_1::number_of_parameters()
            }
            LlmPreset::Phi3Medium4kInstruct => phi_3_medium_4k_instruct::number_of_parameters(),
            LlmPreset::Phi3Mini4kInstruct => phi_3_mini_4k_instruct::number_of_parameters(),
        };
        base_parameters as f64 * 1_000_000_000.0
    }

    pub fn from_model_id(model_id: &str) -> Self {
        let model_id = if model_id.ends_with("-GGUF") {
            model_id.trim_end_matches("-GGUF")
        } else {
            model_id
        };
        match model_id.to_lowercase() {
            x if x == "meta-llama-3-8b-instruct" => LlmPreset::Llama3_8bInstruct,
            x if x == "meta-llama-3-70b-instruct" => LlmPreset::Llama3_70bInstruct,
            x if x == "mistral-7b-instruct-v0.3" => LlmPreset::Mistral7bInstructV0_3,
            x if x == "mixtral-8x7b-instruct-v0.1" => LlmPreset::Mixtral8x7bInstructV0_1,
            x if x == "phi-3-medium-4k-instruct" => LlmPreset::Phi3Medium4kInstruct,
            x if x == "phi-3-mini-4k-instruct" => LlmPreset::Phi3Mini4kInstruct,
            _ => panic!("Model ID not found!"),
        }
    }
}

// Estimates from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/70c0241bc90e8025218a8d9667346aa72f60f472/LLMVRAMCalculator/LLMVRAMCalculator.py#L6

// _GGUF_QUANTS = {
//     "Q2_K": 3.35,
//     "Q3_K_S": 3.5,
//     "Q3_K_M": 3.91,
//     "Q3_K_L": 4.27,
//     "Q4_0": 4.55,
//     "Q4_K_S": 4.58,
//     "Q4_K_M": 4.85,
//     "Q5_0": 5.54,
//     "Q5_K_S": 5.54,
//     "Q5_K_M": 5.69,
//     "Q6_K": 6.59,
//     "Q8_0": 8.5,
// }

pub const Q8: f64 = 8.5;
pub const Q7: f64 = 7.5; // IDK
pub const Q6: f64 = 6.59;
pub const Q5: f64 = 5.90;
pub const Q4: f64 = 4.85;
pub const Q3: f64 = 3.91;
pub const Q2: f64 = 3.35;
pub const Q1: f64 = 2.0; // IDK

pub fn quantization_from_vram(base_model_bytes: f64, vram_gb: u32, context_overhead: f64) -> u8 {
    let cuda_overhead_gb = 0.000; // I don't think this is required.

    let memory_gb = vram_gb as f64 - cuda_overhead_gb - context_overhead;
    let memory_bytes = memory_gb * 1024.0 * 1024.0 * 1024.0;

    // let model_bytes = num_params * bpw / 8
    let estimate_quantized_model_size =
        |base_model_bytes: f64, q_bits: f64| base_model_bytes * q_bits / 8.0;

    match memory_bytes {
        x if x >= estimate_quantized_model_size(base_model_bytes, Q8) => 8,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q7) => 7,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q6) => 6,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q5) => 5,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q4) => 4,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q3) => 3,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q2) => 2,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q1) => 1,
        _ => panic!("Not enough VRAM!"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load() -> Result<()> {
        let model = PresetModelBuilder::default()
            .llama_3_8b_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        let model = PresetModelBuilder::default()
            .llama_3_70b_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        let model = PresetModelBuilder::default()
            .mistral_7b_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        let model = PresetModelBuilder::default()
            .mixtral_8x7b_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        let model = PresetModelBuilder::default()
            .phi_3_medium_4k_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        let model = PresetModelBuilder::default()
            .phi_3_mini_4k_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:#?}", model);

        Ok(())
    }
}
