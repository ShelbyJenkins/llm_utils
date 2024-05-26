use super::{quantization_from_vram, OpenSourceModelType};
use crate::hf_loader::HuggingFaceLoader;
use anyhow::{anyhow, Result};
use gguf_rs::get_gguf_container;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug)]
pub struct GGUFModel {
    pub model_id: String,
    pub local_model_path: String,
    pub metadata: GGUFMetadata,
}

#[derive(Default)]
pub struct GGUFModelBuilder {
    pub hf_token: Option<String>,
    pub open_source_model_type: OpenSourceModelType,
    pub quantization_from_vram_gb: u32,
    pub use_ctx_size: u32,
    pub quant_file_url: Option<String>,
    hf_loader: Option<HuggingFaceLoader>,
}

impl GGUFModelBuilder {
    pub fn new(hf_token: Option<String>) -> Self {
        Self {
            hf_token,
            use_ctx_size: 4444,
            quantization_from_vram_gb: 12,
            ..Default::default()
        }
    }

    pub fn hf_token(mut self, hf_token: &str) -> Self {
        self.hf_token = Some(hf_token.to_string());
        self
    }

    pub fn use_ctx_size(mut self, use_ctx_size: u32) -> Self {
        self.use_ctx_size = use_ctx_size;
        self
    }

    pub fn from_quant_file_url(mut self, quant_file_url: &str) -> Self {
        self.quant_file_url = Some(quant_file_url.to_string());
        self
    }

    pub fn mistral_7b_instruct(mut self) -> Self {
        self.open_source_model_type = OpenSourceModelType::Mistral7bInstructV0_3;
        self
    }

    pub fn mixtral_8x7b_instruct(mut self) -> Self {
        self.open_source_model_type = OpenSourceModelType::Mixtral8x7bInstruct;
        self
    }

    pub fn mixtral_8x22b_instruct(mut self) -> Self {
        self.open_source_model_type = OpenSourceModelType::Mixtral8x22bInstruct;
        self
    }

    pub fn llama_3_70b_instruct(mut self) -> Self {
        self.open_source_model_type = OpenSourceModelType::Llama3_70bInstruct;
        self
    }

    pub fn llama_3_8b_instruct(mut self) -> Self {
        self.open_source_model_type = OpenSourceModelType::Llama3_8bInstruct;
        self
    }

    pub fn quantization_from_vram(mut self, quantization_from_vram_gb: u32) -> Self {
        self.quantization_from_vram_gb = quantization_from_vram_gb;
        self
    }

    pub async fn load(&mut self) -> Result<GGUFModel> {
        let local_model_filename = if let Some(quant_file_url) = &self.quant_file_url {
            let (_, repo_id, gguf_model_filename) =
                HuggingFaceLoader::parse_full_model_url(quant_file_url);
            self.hf_loader =
                Some(HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(&repo_id));
            self.hf_loader
                .as_ref()
                .unwrap()
                .load_file(&gguf_model_filename)
                .await?
        } else {
            let model_id = self.open_source_model_type.model_id();
            let repo_id = self.open_source_model_type.gguf_repo_id();
            self.hf_loader =
                Some(HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(&repo_id));
            let quantization_bits = quantization_from_vram(
                self.open_source_model_type.number_of_parameters(),
                self.quantization_from_vram_gb,
                estimate_context_length_max_memory_requirements(self.use_ctx_size),
            );

            self.try_load(&model_id, quantization_bits).await?
        };

        let local_model_path = HuggingFaceLoader::canonicalize_local_path(local_model_filename)?;

        Ok(GGUFModel {
            model_id: self.open_source_model_type.model_id(),
            metadata: GGUFMetadata::new(&local_model_path)?,
            local_model_path,
        })
    }

    async fn try_load(&self, model_id: &str, mut quantization_bits: u8) -> Result<PathBuf> {
        let mut gguf_model_filename;
        let local_model_filename;

        loop {
            gguf_model_filename = Self::build_gguf_filename(model_id, quantization_bits, "_K_M");

            match self
                .hf_loader
                .as_ref()
                .unwrap()
                .load_file(&gguf_model_filename)
                .await
            {
                Ok(filename) => {
                    local_model_filename = filename;
                    break;
                }
                Err(_) => {
                    gguf_model_filename =
                        Self::build_gguf_filename(model_id, quantization_bits, "_K");

                    match self
                        .hf_loader
                        .as_ref()
                        .unwrap()
                        .load_file(&gguf_model_filename)
                        .await
                    {
                        Ok(filename) => {
                            local_model_filename = filename;
                            break;
                        }
                        Err(_) => {
                            if quantization_bits == 1 {
                                return Err(anyhow!(
                                    "Failed to load model with any quantization level"
                                ));
                            }
                            quantization_bits -= 1;
                        }
                    }
                }
            }
        }

        Ok(local_model_filename)
    }

    // In theory we could be 'clever' and try all permutations of quantization types,
    // but since this will be depreciated once we have ISQ support from mistral-rs we won't bother.
    fn build_gguf_filename(model_id: &str, quantization_bits: u8, quant_type: &str) -> String {
        let mut gguf_filename = model_id.to_string();
        gguf_filename.push('.');
        match quantization_bits {
            1 => gguf_filename.push_str("Q1"),
            2 => gguf_filename.push_str("Q2"),
            3 => gguf_filename.push_str("Q3"),
            4 => gguf_filename.push_str("Q4"),
            5 => gguf_filename.push_str("Q5"),
            6 => gguf_filename.push_str("Q6"),
            7 => gguf_filename.push_str("Q7"),
            8 => gguf_filename.push_str("Q8"),
            _ => panic!("Invalid quantization bits"),
        };

        gguf_filename.push_str(quant_type);
        gguf_filename.push_str(".gguf");
        gguf_filename
    }
}

// We have to use some estimates because the config data is in the GGUF file iteself.
// This is definetly not accurate but it's better than nothing.
// For example llama3 vocab size is 128k but we use 32k.
// In some quick tests it does work though.
fn estimate_context_length_max_memory_requirements(use_ctx_size: u32) -> f64 {
    let hidden_size = use_ctx_size as u64;
    let num_attention_heads = 32;
    let intermediate_size = 14336;
    let vocab_size = 32768;
    let max_position_embeddings = use_ctx_size as u64;

    let attention_params = hidden_size * (num_attention_heads * 3);
    let ffn_params = hidden_size * intermediate_size * 2;
    let embedding_params = vocab_size * hidden_size * 2;
    let position_embedding_params = max_position_embeddings * hidden_size;

    let total_params = attention_params + ffn_params + embedding_params + position_embedding_params;

    // let bytes_per_param = match config.torch_dtype.as_str() {
    //     "float32" => 4,
    //     "float16" | "bfloat16" => 2,
    //     _ => panic!("Unsupported data type"),
    // };

    let memory_requirements_bytes = total_params * 2;

    memory_requirements_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[derive(Debug, Deserialize, Clone)]
pub struct GGUFMetadata {
    pub embedding_length: u32,    // hidden_size
    pub head_count: u32,          //num_attention_heads
    pub feed_forward_length: u32, // intermediate_size
    pub context_length: u32,      // max_position_embeddings
    pub chat_template: String,
}

impl GGUFMetadata {
    pub fn new(local_model_path: &str) -> Result<Self> {
        let gguf_model: gguf_rs::GGUFModel = get_gguf_container(local_model_path)?.decode()?;
        let metadata = gguf_model.metadata();

        Ok(Self {
            embedding_length: metadata
                .get("llama.embedding_length")
                .ok_or_else(|| anyhow!("llama.embedding_length not found in metadata"))?
                .as_u64()
                .ok_or_else(|| anyhow!("llama.embedding_length is not a valid u64"))?
                .try_into()
                .unwrap(),
            head_count: metadata
                .get("llama.attention.head_count")
                .ok_or_else(|| anyhow!("llama.attention.head_count not found in metadata"))?
                .as_u64()
                .ok_or_else(|| anyhow!("llama.attention.head_count is not a valid u64"))?
                .try_into()
                .unwrap(),
            feed_forward_length: metadata
                .get("llama.feed_forward_length")
                .ok_or_else(|| anyhow!("llama.feed_forward_length not found in metadata"))?
                .as_u64()
                .ok_or_else(|| anyhow!("llama.feed_forward_length is not a valid u64"))?
                .try_into()
                .unwrap(),
            context_length: metadata
                .get("llama.context_length")
                .ok_or_else(|| anyhow!("llama.context_length not found in metadata"))?
                .as_u64()
                .ok_or_else(|| anyhow!("llama.context_length is not a valid u64"))?
                .try_into()
                .unwrap(),
            chat_template: metadata
                .get("tokenizer.chat_template")
                .ok_or_else(|| anyhow!("tokenizer.chat_template not found in metadata"))?
                .as_str()
                .ok_or_else(|| anyhow!("tokenizer.chat_template is not a valid string"))?
                .to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load() -> Result<()> {
        let model = GGUFModelBuilder::default()
            .mixtral_8x7b_instruct()
            .quantization_from_vram(48)
            .load()
            .await?;

        println!("{:?}", model);
        Ok(())
    }
}
