pub mod chat_template_format;
pub mod concat;
mod local_content;
pub mod prompt_message;
pub mod token_count;

use crate::models::{
    api_model::ApiLlm,
    open_source_model::{OsLlm, OsLlmChatTemplate},
};
use anyhow::{anyhow, Result};
use chat_template_format::apply_chat_template;
pub use concat::{PromptConcatenator, PromptConcatenatorTrait};
use local_content::load_content_path;
use prompt_message::{add_assistant_message, add_system_message, add_user_message, build_messages};
pub use prompt_message::{PromptMessage, PromptMessageType};
use std::{collections::HashMap, path::PathBuf};

#[derive(Clone)]
pub enum PromptFormat {
    ChatTemplate(OsLlm),
    OpenAi(ApiLlm),
    Anthropic(ApiLlm),
}

impl LlmPrompt {
    pub fn new_from_os_llm(os_llm: &OsLlm) -> LlmPrompt {
        LlmPrompt::new(PromptFormat::ChatTemplate(os_llm.clone()))
    }

    pub fn new_from_openai_llm(openai_llm: &ApiLlm) -> LlmPrompt {
        LlmPrompt::new(PromptFormat::OpenAi(openai_llm.clone()))
    }

    pub fn new_from_openai_model_id(model_id: &str) -> Result<LlmPrompt> {
        Ok(LlmPrompt::new(PromptFormat::OpenAi(
            ApiLlm::openai_model_from_model_id(model_id),
        )))
    }

    pub fn new_from_anthropic_llm(anthropic_llm: &ApiLlm) -> LlmPrompt {
        LlmPrompt::new(PromptFormat::Anthropic(anthropic_llm.clone()))
    }

    pub fn new_from_anthropic_model_id(model_id: &str) -> LlmPrompt {
        LlmPrompt::new(PromptFormat::Anthropic(
            ApiLlm::anthropic_model_from_model_id(model_id),
        ))
    }
}

impl std::fmt::Display for PromptFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptFormat::ChatTemplate(_) => write!(f, "PromptFormat::ChatTemplate"),
            PromptFormat::OpenAi(_) => write!(f, "PromptFormat::OpenAi"),
            PromptFormat::Anthropic(_) => write!(f, "PromptFormat::Anthropic"),
        }
    }
}

#[derive(Clone)]
pub struct LlmPrompt {
    pub built_chat_template_prompt: Option<String>,
    pub built_prompt_as_tokens: Option<Vec<u32>>,
    pub built_openai_prompt: Option<Vec<HashMap<String, String>>>,
    pub concatenator: PromptConcatenator,
    pub messages: Vec<PromptMessage>,
    pub prompt_format: PromptFormat,
    pub total_prompt_tokens: Option<u32>,
    generation_prefix: Option<String>,
}

impl LlmPrompt {
    fn new(prompt_format: PromptFormat) -> Self {
        Self {
            messages: Vec::new(),
            prompt_format,
            concatenator: PromptConcatenator::SingleNewline,
            built_chat_template_prompt: None,
            built_openai_prompt: None,
            built_prompt_as_tokens: None,
            total_prompt_tokens: None,
            generation_prefix: None,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        self.precheck_build()?;
        self.build_backend(None)?;
        Ok(())
    }

    pub fn build_with_generation_prefix<T: AsRef<str>>(
        &mut self,
        generation_prefix: T,
    ) -> Result<()> {
        self.precheck_build()?;
        if self.generation_prefix.is_none()
            || self.generation_prefix.as_deref() != Some(generation_prefix.as_ref())
        {
            self.clear_built_prompt();
            self.generation_prefix = Some(generation_prefix.as_ref().to_string());
        }
        self.build_backend(Some(generation_prefix.as_ref()))
    }

    pub fn build_final(&mut self) -> Result<()> {
        self.clear_built_prompt();
        self.build_backend(None)?;
        Ok(())
    }

    pub fn add_system_message(&mut self) -> &mut PromptMessage {
        if matches!(self.prompt_format, PromptFormat::ChatTemplate(_)) {
            panic!("Cannot add system message to chat template prompt.");
        }
        add_system_message(&mut self.messages, &self.concatenator);
        self.clear_built_prompt();
        self.messages.last_mut().unwrap()
    }

    pub fn add_user_message(&mut self) -> &mut PromptMessage {
        add_user_message(&mut self.messages, &self.concatenator);
        self.clear_built_prompt();
        self.messages.last_mut().unwrap()
    }

    pub fn add_assistant_message(&mut self) -> &mut PromptMessage {
        add_assistant_message(&mut self.messages, &self.concatenator);
        self.clear_built_prompt();
        self.messages.last_mut().unwrap()
    }

    pub fn reset_prompt(&mut self) {
        self.messages.clear();
        self.clear_built_prompt();
    }

    pub fn clear_built_prompt(&mut self) {
        self.built_openai_prompt = None;
        self.built_chat_template_prompt = None;
        self.built_prompt_as_tokens = None;
        self.total_prompt_tokens = None;
    }

    pub fn requires_build(&self) -> bool {
        match self.prompt_format {
            PromptFormat::ChatTemplate(_) => {
                self.built_chat_template_prompt.is_none()
                    || self.messages.iter().any(|m| m.requires_build())
            }
            _ => {
                self.built_openai_prompt.is_none()
                    || self.messages.iter().any(|m| m.requires_build())
            }
        }
    }

    pub fn total_prompt_tokens(&mut self) -> Result<u32> {
        if let Some(total_prompt_tokens) = self.total_prompt_tokens {
            Ok(total_prompt_tokens)
        } else {
            Err(anyhow!("total_prompt_tokens is None."))
        }
    }

    fn precheck_build(&self) -> Result<()> {
        if let Some(last) = self.messages.last() {
            if last.message_type == PromptMessageType::Assistant {
                Err(anyhow!(
                    "Cannot build prompt when the current inference message is PromptMessageType::Assistant"
                ))
            } else if last.message_type == PromptMessageType::System {
                Err(anyhow!("Cannot build prompt when the current inference message is PromptMessageType::System"))
            } else {
                Ok(())
            }
        } else {
            Err(anyhow!("Cannot build prompt when there are no messages."))
        }
    }

    fn build_backend(&mut self, generation_prefix: Option<&str>) -> Result<()> {
        if !self.requires_build() {
            return Ok(());
        }
        self.clear_built_prompt();
        let prompt_messages = build_messages(&mut self.messages);

        match &self.prompt_format {
            PromptFormat::ChatTemplate(model) => {
                let mut built_chat_template_prompt =
                    apply_chat_template(&prompt_messages.iter().collect(), &model.chat_template);
                if let Some(generation_prefix) = generation_prefix {
                    built_chat_template_prompt
                        .push_str(model.chat_template.base_generation_prefix.as_ref().unwrap());
                    built_chat_template_prompt.push_str(generation_prefix);
                }
                if let Some(tokenizer) = &model.tokenizer {
                    let built_prompt_as_tokens = tokenizer.tokenize(&built_chat_template_prompt);
                    self.built_prompt_as_tokens = Some(built_prompt_as_tokens);
                };
                self.built_chat_template_prompt = Some(built_chat_template_prompt);
            }
            _ => {
                self.built_openai_prompt = Some(prompt_messages);
            }
        }
        self.total_prompt_tokens = self.count_prompt_tokens();
        Ok(())
    }

    fn count_prompt_tokens(&self) -> Option<u32> {
        match &self.prompt_format {
            PromptFormat::ChatTemplate(_) => {
                if let Some(built_prompt_as_tokens) = &self.built_prompt_as_tokens {
                    Some(built_prompt_as_tokens.len() as u32)
                } else {
                    panic!("Chat template for PromptFormat::ChatTemplate should be set by now.")
                }
            }
            PromptFormat::OpenAi(model) => {
                if let Some(built_openai_prompt) = &self.built_openai_prompt {
                    Some(model.openai_token_count_of_prompt(built_openai_prompt))
                } else {
                    panic!("OpenAI prompt for PromptFormat::OpenAi should be set by now.")
                }
            }
            PromptFormat::Anthropic(model) => {
                if let Some(built_openai_prompt) = &self.built_openai_prompt {
                    Some(model.anthropic_token_count_of_prompt(built_openai_prompt))
                } else {
                    panic!("Anthropic prompt for PromptFormat::Anthropic should be set by now.")
                }
            }
        }
    }
}

impl PromptConcatenatorTrait for LlmPrompt {
    fn concate_mut(&mut self) -> &mut PromptConcatenator {
        &mut self.concatenator
    }
    fn clear_built(&mut self) {
        self.clear_built_prompt();
    }
}

impl std::fmt::Display for LlmPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        let prompt_format = match self.prompt_format {
            PromptFormat::ChatTemplate(_) => "ChatTemplate",
            PromptFormat::OpenAi(_) | PromptFormat::Anthropic(_) => "OpenAi Compatible",
        };
        writeln!(f, "\x1b[1m\x1b[34mLlmPrompt\x1b[0m: {prompt_format}")?;
        writeln!(f)?;
        for message in self.messages.iter() {
            writeln!(f, "{}", message)?;
        }
        if let Some(built_chat_template_prompt) = &self.built_chat_template_prompt {
            writeln!(
                f,
                "built_chat_template_prompt:\n\n{:?}",
                built_chat_template_prompt
            )?;
            writeln!(f)?;
        };

        if let Some(built_openai_prompt) = &self.built_openai_prompt {
            writeln!(f, "built_openai_prompt:\n\n{:?}", built_openai_prompt)?;
            writeln!(f)?;
        };

        if let Some(total_prompt_tokens) = &self.total_prompt_tokens {
            writeln!(f, "total_prompt_tokens: {}", total_prompt_tokens)?;
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat() {
        let model = OsLlm::default();
        let mut prompt = LlmPrompt::new_from_os_llm(&model);

        prompt.add_user_message().set_content("test user content 1");
        prompt
            .add_assistant_message()
            .set_content("test assistant content");
        prompt.add_user_message().set_content("test user content 2");

        prompt.build().unwrap();
        println!("{prompt}",);
        let test_chat = prompt.built_chat_template_prompt.as_ref().unwrap();
        assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|>"
        );
        let token_count = prompt.total_prompt_tokens.unwrap();
        let prompt_as_tokens = prompt.built_prompt_as_tokens.as_ref().unwrap();
        assert_eq!(29, token_count);
        assert_eq!(token_count, prompt_as_tokens.len() as u32);

        prompt
            .build_with_generation_prefix("Generating 12345:")
            .unwrap();
        println!("{prompt}");
        let test_chat = prompt.built_chat_template_prompt.as_ref().unwrap();
        assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerating 12345:"
        );
        let token_count = prompt.total_prompt_tokens.unwrap();
        let prompt_as_tokens = prompt.built_prompt_as_tokens.as_ref().unwrap();
        assert_eq!(38, token_count);
        assert_eq!(token_count, prompt_as_tokens.len() as u32);
    }

    #[test]
    fn test_openai() {
        let mut prompt = LlmPrompt::new_from_openai_llm(&ApiLlm::default());

        prompt
            .add_system_message()
            .set_content("test system content");
        prompt.add_user_message().set_content("test user content 1");
        prompt
            .add_assistant_message()
            .set_content("test assistant content");
        prompt.add_user_message().set_content("test user content 2");
        prompt.build().unwrap();
        println!("{prompt}");
        let test_openai = prompt.built_openai_prompt.unwrap();
        let result_openai = vec![
            HashMap::from([
                ("content".to_string(), "test system content".to_string()),
                ("role".to_string(), "system".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test user content 1".to_string()),
                ("role".to_string(), "user".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test assistant content".to_string()),
                ("role".to_string(), "assistant".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test user content 2".to_string()),
                ("role".to_string(), "user".to_string()),
            ]),
        ];
        assert_eq!(test_openai, result_openai);
        let token_count = prompt.total_prompt_tokens.unwrap();
        assert_eq!(31, token_count);
    }
}
