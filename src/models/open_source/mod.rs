use crate::{hf_loader::HuggingFaceLoader, tokenizer::LlmTokenizer};
pub use anyhow::Result;
pub use safe_tensors::{chat_template_from_local, model_config_json_from_local, OsLlmConfigJson};
use serde::Deserialize;
use std::{
    fmt::{self, Debug},
    path::PathBuf,
};
pub mod gguf;
pub mod safe_tensors;
use anyhow::anyhow;
pub mod preset;
pub use gguf::GGUFModelBuilder;
pub use preset::{LlmPreset, PresetModelBuilder};
pub use safe_tensors::SafeTensorsModelBuilder;
pub use std::{env, fs::File, io::BufReader};

pub struct OsLlm {
    pub model_id: String,
    pub model_url: String,
    pub local_model_paths: Vec<String>,
    pub model_config_json: OsLlmConfigJson,
    pub chat_template: OsLlmChatTemplate,
    pub tokenizer: Option<LlmTokenizer>,
}

impl fmt::Debug for OsLlm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("OsLlm");
        debug_struct.field("model_id", &self.model_id);
        debug_struct.field("model_url", &self.model_url);
        debug_struct.field("local_model_paths", &self.local_model_paths);
        debug_struct.field("model_config_json", &self.model_config_json);
        debug_struct.field("chat_template", &self.chat_template);
        debug_struct.field("tokenizer", &self.tokenizer);
        debug_struct.finish()
    }
}

#[derive(Deserialize, Debug)]
pub struct OsLlmChatTemplate {
    pub chat_template: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
}
