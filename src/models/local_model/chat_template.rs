use super::gguf::gguf_metadata::GgufMetadata;
use anyhow::Context;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct LlmChatTemplate {
    pub chat_template: String,
    pub bos_token: String,
    pub eos_token: String,
    pub unk_token: Option<String>,
    pub base_generation_prefix: Option<String>,
}

impl LlmChatTemplate {
    pub fn chat_template_from_local(
        tokenizer_config_json_path: &std::path::PathBuf,
    ) -> crate::Result<Self> {
        let file = std::fs::File::open(tokenizer_config_json_path)?;
        let reader = std::io::BufReader::new(file);
        let mut chat_template: LlmChatTemplate = serde_json::from_reader(reader)?;
        chat_template.set_generation_prefix()?;
        Ok(chat_template)
    }

    pub fn chat_template_from_gguf(metadata: &GgufMetadata) -> crate::Result<Self> {
        let chat_template: String = if let Some(chat_template) =
            metadata.get_option_value(&["tokenizer"], "chat_template")?
        {
            chat_template
        } else {
            crate::bail!("chat_template not found in metadata");
        };
        let tokens: Vec<String> = metadata.get_value(&["tokenizer", "ggml"], "tokens")?;
        let bos_token_id: u32 = metadata.get_value(&["tokenizer", "ggml"], "bos_token_id")?;
        let bos_token = tokens
            .get(bos_token_id as usize)
            .map(ToString::to_string)
            .with_context(|| format!("Token not found for ID: {}", bos_token_id))?;

        let eos_token_id: u32 = metadata.get_value(&["tokenizer", "ggml"], "eos_token_id")?;
        let eos_token = tokens
            .get(eos_token_id as usize)
            .map(ToString::to_string)
            .with_context(|| format!("Token not found for ID: {}", eos_token_id))?;

        let unk_token_id: Option<u32> =
            metadata.get_option_value(&["tokenizer", "ggml"], "unk_token_id")?;
        let unk_token = if let Some(unk_token_id) = unk_token_id {
            Some(
                tokens
                    .get(unk_token_id as usize)
                    .map(ToString::to_string)
                    .with_context(|| format!("Token not found for ID: {}", unk_token_id))?,
            )
        } else {
            None
        };

        let mut chat_template = LlmChatTemplate {
            chat_template,
            bos_token,
            eos_token,
            unk_token,
            base_generation_prefix: None,
        };
        chat_template.set_generation_prefix()?;
        Ok(chat_template)
    }

    fn set_generation_prefix(&mut self) -> crate::Result<()> {
        let user_message_1 = HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), "test_user_message_1".to_string()),
        ]);
        let assistant_message_1 = HashMap::from([
            ("role".to_string(), "assistant".to_string()),
            (
                "content".to_string(),
                "test_assistant_message_1".to_string(),
            ),
        ]);

        let message_1 = crate::prompting::chat_template_prompt::apply_chat_template(
            &vec![user_message_1.clone()],
            &self.chat_template,
            &self.bos_token,
            &self.eos_token,
            self.unk_token.as_deref(),
        );
        let message_1 = message_1
            .trim_end_matches(self.eos_token.as_str())
            .to_owned();
        let message_2 = crate::prompting::chat_template_prompt::apply_chat_template(
            &vec![user_message_1, assistant_message_1],
            &self.chat_template,
            &self.bos_token,
            &self.eos_token,
            self.unk_token.as_deref(),
        );

        // Find the point where the outputs start to differ
        let diff_index = message_1
            .chars()
            .zip(message_2.chars())
            .position(|(a, b)| a != b)
            .unwrap_or(message_1.len());

        // Extract the differing part
        let diff_part = &message_2[diff_index..];

        // Find the start of the assistant content
        if let Some(content_index) = diff_part.find("test_assistant_message_1") {
            // The prefix is everything before the content
            self.base_generation_prefix = Some(
                diff_part[..content_index]
                    .trim_start_matches(self.eos_token.as_str())
                    .to_string(),
            );
        } else {
            crate::bail!("Error finding base_generation_prefix");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::models::local_model::{gguf::GgufLoader, LlmPresetTrait};

    #[test]
    fn test_base_generation_prefix() {
        let model = GgufLoader::default()
            .llama3_1_8b_instruct()
            .available_vram(48)
            .load()
            .unwrap();
        assert_eq!(
            Some("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            model.chat_template.base_generation_prefix.as_deref()
        );
        let model = GgufLoader::default()
            .mistral7b_instruct_v0_3()
            .available_vram(48)
            .load()
            .unwrap();
        assert_eq!(
            Some(""),
            model.chat_template.base_generation_prefix.as_deref()
        );
        let model = GgufLoader::default()
            .phi3_5_mini_instruct()
            .available_vram(48)
            .load()
            .unwrap();
        assert_eq!(
            Some("<|assistant|>\n"),
            model.chat_template.base_generation_prefix.as_deref()
        );
    }
}
