use super::preset::LlmPreset;
use anyhow::Result;
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::PathBuf};

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct OsLlmChatTemplate {
    pub chat_template: String,
    pub chat_template_path: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub base_generation_prefix: Option<String>,
}

impl Default for OsLlmChatTemplate {
    fn default() -> Self {
        LlmPreset::default()
            .chat_template()
            .expect("Failed to get default chat template")
    }
}

pub fn chat_template_from_local(tokenizer_config_json_path: &PathBuf) -> Result<OsLlmChatTemplate> {
    let file = File::open(tokenizer_config_json_path)?;
    let reader = BufReader::new(file);
    let mut chat_template: OsLlmChatTemplate = serde_json::from_reader(reader)?;
    chat_template.chat_template_path =
        Some(tokenizer_config_json_path.to_string_lossy().to_string());
    Ok(chat_template)
}
