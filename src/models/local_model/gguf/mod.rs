use super::{
    hf_loader::HuggingFaceLoader,
    model_metadata::{model_metadata_from_gguf, model_metadata_from_local, LocalLlmMetadata},
    HfTokenTrait,
    LlmChatTemplate,
    LlmPresetLoader,
    LlmPresetTrait,
    LocalLlmModel,
};
use crate::{models::LlmModelBase, tokenizer::LlmTokenizer};
use gguf_metadata::GgufMetadata;
use gguf_tokenizer::convert_gguf_to_hf_tokenizer;
use vram::{estimate_context_size, quantization_from_vram};

pub mod gguf_file;
pub mod gguf_metadata;
pub mod gguf_tokenizer;
pub mod vram;

const DEFAULT_CTX_SIZE: u32 = 4096;

#[derive(Default)]
pub struct GgufLoader {
    pub preset_loader: LlmPresetLoader,
    pub local_quant_file_path: Option<std::path::PathBuf>,
    pub local_config_path: Option<std::path::PathBuf>,
    pub local_tokenizer_path: Option<std::path::PathBuf>,
    pub local_tokenizer_config_path: Option<std::path::PathBuf>,
    pub hf_quant_file_url: Option<String>,
    pub hf_config_repo_id: Option<String>,
    pub hf_tokenizer_repo_id: Option<String>,
    pub hf_tokenizer_config_repo_id: Option<String>,
    pub hf_loader: super::hf_loader::HuggingFaceLoader,
    gguf_metadata: Option<GgufMetadata>,
    pub model_id: Option<String>,
    pub local_model_path: Option<std::path::PathBuf>,
}

impl GgufLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self) -> crate::Result<LocalLlmModel> {
        let local_model_path =
            if self.local_quant_file_path.is_some() || self.hf_quant_file_url.is_some() {
                self.local_model_path()?
            } else {
                self.preset_model_path()?
            };
        let model_metadata = self.model_metadata()?;
        Ok(LocalLlmModel {
            model_base: LlmModelBase {
                model_id: self
                    .model_id
                    .clone()
                    .expect("model_id should have been set by this point"),
                context_length: model_metadata.max_position_embeddings as u32,
                max_tokens_output: model_metadata.max_position_embeddings as u32,
                tokenizer: self.tokenizer()?,
            },
            model_metadata,
            chat_template: self.chat_template()?,
            local_model_path,
        })
    }

    fn local_model_path(&mut self) -> crate::Result<std::path::PathBuf> {
        if self.local_quant_file_path.is_some() && self.hf_quant_file_url.is_some() {
            crate::bail!(
                "both local_quant_file_path and hf_quant_file_url can not be set at the same time"
            );
        };
        if let Some(hf_quant_file_url) = self.hf_quant_file_url.as_ref() {
            let (model_id, repo_id, gguf_model_filename) =
                HuggingFaceLoader::parse_full_model_url(hf_quant_file_url);

            if self.model_id.is_none() {
                self.model_id = Some(model_id);
            }

            self.local_model_path = Some(HuggingFaceLoader::canonicalize_local_path(
                self.hf_loader.load_file(gguf_model_filename, repo_id)?,
            )?);
        } else if let Some(local_quant_file_path) = self.local_quant_file_path.as_ref() {
            self.local_model_path = Some(local_quant_file_path.to_owned());
            if self.model_id.is_none() {
                self.model_id = Some(local_quant_file_path.to_string_lossy().to_string());
            }
        };
        self.local_model_path.clone().ok_or_else(|| {
            crate::anyhow!("one of local_quant_file_path and hf_quant_file_url can must be set")
        })
    }

    fn preset_model_path(&mut self) -> crate::Result<std::path::PathBuf> {
        // Presumes a batch_size of 512
        let q_bits = quantization_from_vram(
            self.preset_loader.llm_preset.number_of_parameters(),
            self.preset_loader.available_vram,
            estimate_context_size(
                &self.preset_loader.llm_preset.model_metadata()?,
                self.preset_loader.use_ctx_size.unwrap_or(DEFAULT_CTX_SIZE) as u64,
                512,
            ),
        );
        let local_model_filename = self.hf_loader.load_file(
            self.preset_loader
                .llm_preset
                .f_name_for_q_bits(q_bits)
                .expect("Invalid quantization bits"),
            self.preset_loader.llm_preset.gguf_repo_id(),
        )?;
        self.model_id = Some(self.preset_loader.llm_preset.model_id());
        if self.local_config_path.is_none() {
            self.local_config_path = Some(self.preset_loader.llm_preset.config_json_path());
        }
        if self.local_tokenizer_path.is_none() {
            self.local_tokenizer_path = self.preset_loader.llm_preset.tokenizer_path();
        }
        if self.local_tokenizer_config_path.is_none() {
            self.local_tokenizer_config_path =
                self.preset_loader.llm_preset.tokenizer_config_path();
        }
        let local_model_path = HuggingFaceLoader::canonicalize_local_path(local_model_filename)?;
        self.local_model_path = Some(local_model_path.clone());
        Ok(local_model_path)
    }

    fn gguf_metadata(&mut self) -> crate::Result<&GgufMetadata> {
        if self.gguf_metadata.is_none() {
            self.gguf_metadata = Some(GgufMetadata::from_path(
                self.local_model_path
                    .as_ref()
                    .expect("local_model_path should have been set by this point"),
            )?);
        }
        Ok(self.gguf_metadata.as_ref().unwrap())
    }

    fn model_metadata(&mut self) -> crate::Result<LocalLlmMetadata> {
        if self.local_config_path.is_none() {
            if let Some(hf_config_repo_id) = self.hf_config_repo_id.as_ref() {
                let local_config_path =
                    self.hf_loader.load_file("config.json", hf_config_repo_id)?;
                self.local_config_path = Some(local_config_path);
            }
        };
        if let Some(local_config_path) = &self.local_config_path {
            model_metadata_from_local(&std::path::PathBuf::from(local_config_path))
        } else {
            model_metadata_from_gguf(self.gguf_metadata()?)
        }
    }

    fn tokenizer(&mut self) -> crate::Result<std::sync::Arc<LlmTokenizer>> {
        if self.local_tokenizer_path.is_none() {
            if let Some(hf_tokenizer_repo_id) = self.hf_tokenizer_repo_id.as_ref() {
                let local_tokenizer_path = self
                    .hf_loader
                    .load_file("tokenizer.json", hf_tokenizer_repo_id)?;
                self.local_tokenizer_path = Some(local_tokenizer_path);
            }
        };
        if let Some(local_tokenizer_path) = &self.local_tokenizer_path {
            Ok(std::sync::Arc::new(LlmTokenizer::new_from_tokenizer_json(
                local_tokenizer_path,
            )?))
        } else {
            let tokenizer = convert_gguf_to_hf_tokenizer(self.gguf_metadata()?)?;
            Ok(std::sync::Arc::new(LlmTokenizer::new_from_tokenizer(
                tokenizer,
            )?))
        }
    }

    fn chat_template(&mut self) -> crate::Result<LlmChatTemplate> {
        if self.local_tokenizer_config_path.is_none() {
            if let Some(hf_tokenizer_config_repo_id) = self.hf_tokenizer_config_repo_id.as_ref() {
                let local_tokenizer_config_path = self
                    .hf_loader
                    .load_file("tokenizer_config.json", hf_tokenizer_config_repo_id)?;
                self.local_tokenizer_config_path = Some(local_tokenizer_config_path);
            }
        };
        if let Some(local_tokenizer_config_path) = &self.local_tokenizer_config_path {
            LlmChatTemplate::chat_template_from_local(local_tokenizer_config_path)
        } else {
            LlmChatTemplate::chat_template_from_gguf(self.gguf_metadata()?)
        }
    }
}

impl GgufLoaderTrait for GgufLoader {
    fn gguf_loader(&mut self) -> &mut GgufLoader {
        self
    }
}

impl HfTokenTrait for GgufLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_loader.hf_token_env_var
    }
}

impl LlmPresetTrait for GgufLoader {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        &mut self.preset_loader
    }
}

pub trait GgufLoaderTrait {
    fn gguf_loader(&mut self) -> &mut GgufLoader;

    /// Sets the model id for the model config. Used for display purposes and debugging.
    /// Optional because this can be loaded from the URL, file path, or preset.
    fn model_id<S: Into<String>>(&mut self, model_id: S) -> &mut Self {
        self.gguf_loader().model_id = Some(model_id.into());
        self
    }

    /// Sets the local path to the quantized model file.
    /// Use the /full/path/and/filename.gguf
    fn local_quant_file_path<S: Into<std::path::PathBuf>>(
        &mut self,
        local_quant_file_path: S,
    ) -> &mut Self {
        self.gguf_loader().local_quant_file_path = Some(local_quant_file_path.into());
        self
    }

    /// Sets the local path to the model config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_config_path<P: AsRef<std::path::Path>>(&mut self, local_config_path: P) -> &mut Self {
        self.gguf_loader().local_config_path = Some(local_config_path.as_ref().to_owned());
        self
    }

    /// Sets the local path to the tokenizer.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_tokenizer_path<P: AsRef<std::path::Path>>(
        &mut self,
        local_tokenizer_path: P,
    ) -> &mut Self {
        self.gguf_loader().local_tokenizer_path = Some(local_tokenizer_path.as_ref().to_owned());
        self
    }

    /// Sets the local path to the tokenizer_config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn local_tokenizer_config_path<P: AsRef<std::path::Path>>(
        &mut self,
        local_tokenizer_config_path: P,
    ) -> &mut Self {
        self.gguf_loader().local_tokenizer_config_path =
            Some(local_tokenizer_config_path.as_ref().to_owned());
        self
    }

    /// Sets the Hugging Face url to the quantized model file.
    /// The full url to the model on hugging face like:
    /// 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf'
    fn hf_quant_file_url<S: Into<String>>(&mut self, hf_quant_file_url: S) -> &mut Self {
        self.gguf_loader().hf_quant_file_url = Some(hf_quant_file_url.into());
        self
    }

    /// Sets the Hugging Face repo id to the model config.json file.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_config_repo_id<S: Into<String>>(&mut self, hf_config_repo_id: S) -> &mut Self {
        self.gguf_loader().hf_config_repo_id = Some(hf_config_repo_id.into());
        self
    }

    /// Sets the Hugging Face repo id for the tokenizer. This is used for loading the tokenizer.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_tokenizer_repo_id<S: Into<String>>(&mut self, hf_tokenizer_repo_id: S) -> &mut Self {
        self.gguf_loader().hf_tokenizer_repo_id = Some(hf_tokenizer_repo_id.into());
        self
    }

    /// Sets the Hugging Face repo id for the tokenizer config. This is used for loading the chat template.
    /// Optional because this can be loaded from the GGUF file.
    fn hf_tokenizer_config_repo_id<S: Into<String>>(
        &mut self,
        hf_tokenizer_config_repo_id: S,
    ) -> &mut Self {
        self.gguf_loader().hf_tokenizer_config_repo_id = Some(hf_tokenizer_config_repo_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_hf_basic() {
        let model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
            .load()
            .unwrap();

        println!("{:#?}", model);
    }

    #[test]
    fn load_hf_with_config() {
        let model = GgufLoader::default()
        .hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
            .load()
            .unwrap();

        println!("{:#?}", model);
    }

    #[test]
    fn load_local_basic() {
        let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
            .load()
           .unwrap();

        println!("{:#?}", model);
    }

    #[test]
    fn load_local_with_config() {
        let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
        .local_config_path("/workspaces/test/llm_utils/src/models/local_model/preset/llama/llama3_1_8b_instruct/config.json")
            .load()
           .unwrap();

        println!("{:#?}", model);
    }

    #[test]
    fn load_local_with_hf_config() {
        let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
            .load()
           .unwrap();

        println!("{:#?}", model);
    }
}
