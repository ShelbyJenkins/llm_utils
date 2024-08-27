use crate::tokenizer::LlmTokenizer;
use anyhow::{anyhow, Result};
use hf_loader::HuggingFaceLoader;
use model_config_json::OsLlmConfigJson;
use std::{
    env,
    fmt::{self, Debug},
    path::PathBuf,
    sync::Arc,
};

pub mod chat_template;
pub mod gguf;
pub mod hf_loader;
pub mod model_config_json;
pub mod preset;
pub mod vram;

pub use chat_template::OsLlmChatTemplate;
pub use gguf::{LlmGgufLoader, LlmGgufTrait};
pub use hf_loader::HfTokenTrait;
pub use preset::{preset_loader::LlmPresetLoader, LlmPresetTrait};

#[derive(Clone)]
pub struct OsLlm {
    pub model_id: String,
    pub local_model_path: PathBuf,
    pub model_config_json: OsLlmConfigJson,
    pub chat_template: OsLlmChatTemplate,
    pub tokenizer: Option<Arc<LlmTokenizer>>,
}

impl Default for OsLlm {
    fn default() -> Self {
        let mut loader = OsLlmLoader::default();
        loader.preset_loader();
        loader.load().expect("Failed to load LlmPreset")
    }
}

impl fmt::Debug for OsLlm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("OsLlm");
        debug_struct.field("model_id", &self.model_id);
        debug_struct.field("local_model_path", &self.local_model_path);
        debug_struct.field("model_config_json", &self.model_config_json);
        debug_struct.field("chat_template", &self.chat_template);
        debug_struct.field("tokenizer", &self.tokenizer);
        debug_struct.finish()
    }
}

#[derive(Default)]
pub struct OsLlmLoader {
    pub preset_loader: Option<LlmPresetLoader>,
    pub gguf_loader: Option<LlmGgufLoader>,
    pub hf_loader: HuggingFaceLoader,
}

impl OsLlmLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&self) -> Result<OsLlm> {
        if self.preset_loader.is_some() && self.gguf_loader.is_some() {
            panic!("preset_loader and gguf_loader can not be used at the same time");
        };

        if let Some(preset_loader) = self.preset_loader.as_ref() {
            Ok(OsLlm {
                model_id: preset_loader.llm_preset.model_id(),
                model_config_json: preset_loader.llm_preset.model_config_json()?,
                chat_template: preset_loader.llm_preset.chat_template()?,
                local_model_path: preset_loader.local_model_path(&self.hf_loader)?,
                tokenizer: Some(preset_loader.llm_preset.tokenizer()?),
            })
        } else if let Some(gguf_loader) = self.gguf_loader.as_ref() {
            Ok(OsLlm {
                model_id: gguf_loader.model_id(),
                model_config_json: gguf_loader.model_config_json(&self.hf_loader)?,
                chat_template: gguf_loader.chat_template(&self.hf_loader)?,
                local_model_path: gguf_loader.local_model_path(&self.hf_loader)?,
                tokenizer: gguf_loader.tokenizer(&self.hf_loader)?,
            })
        } else {
            panic!("Both preset_loader and gguf_loader can not be None");
        }
    }
}

impl LlmPresetTrait for OsLlmLoader {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        if self.preset_loader.is_none() {
            self.preset_loader = Some(LlmPresetLoader::new());
        }
        self.preset_loader.as_mut().unwrap()
    }
}

impl LlmGgufTrait for OsLlmLoader {
    fn gguf_loader(&mut self) -> &mut LlmGgufLoader {
        if self.gguf_loader.is_none() {
            self.gguf_loader = Some(LlmGgufLoader::new());
        }
        self.gguf_loader.as_mut().unwrap()
    }
}

impl HfTokenTrait for OsLlmLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_loader.hf_token_env_var
    }
}
