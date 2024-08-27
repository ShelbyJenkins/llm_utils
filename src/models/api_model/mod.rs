use crate::tokenizer::LlmTokenizer;
use std::{collections::HashMap, sync::Arc};

pub mod anthropic;
pub mod openai;
pub mod perplexity;

#[derive(Clone)]
pub struct ApiLlm {
    pub model_id: String,
    pub context_length: u32,
    pub cost_per_m_in_tokens: f32,
    pub max_tokens_output: u32,
    pub cost_per_m_out_tokens: f32,
    pub tokens_per_message: u32,
    pub tokens_per_name: Option<i32>,
    pub tokenizer: Arc<LlmTokenizer>,
    pub api_type: ApiLlmType,
}

impl Default for ApiLlm {
    fn default() -> Self {
        Self::gpt_4_o_mini()
    }
}

impl ApiLlm {
    pub fn token_count_of_prompt(&self, prompt: &Vec<HashMap<String, String>>) -> u32 {
        match self.api_type {
            ApiLlmType::OpenAi => self.openai_token_count_of_prompt(prompt),
            ApiLlmType::Anthropic => self.anthropic_token_count_of_prompt(prompt),
        }
    }

    pub fn openai_token_count_of_prompt(&self, prompt: &Vec<HashMap<String, String>>) -> u32 {
        let mut num_tokens = 0;
        for message in prompt {
            num_tokens += self.tokens_per_message;
            num_tokens += self.tokenizer.count_tokens(&message["content"]);

            let tokens_per_name = self
                .tokens_per_name
                .expect("tokens_per_name is required for this model");
            if message["role"].is_empty() {
                if tokens_per_name < 0 {
                    // Handles cases for certain models where name doesn't count towards token count
                    num_tokens -= tokens_per_name.unsigned_abs();
                } else {
                    num_tokens += tokens_per_name as u32;
                }
            }
        }
        num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
        num_tokens
    }

    pub fn anthropic_token_count_of_prompt(&self, prompt: &Vec<HashMap<String, String>>) -> u32 {
        let mut num_tokens: u32 = 0;
        for message in prompt {
            num_tokens += self.tokens_per_message;
            num_tokens += self.tokenizer.count_tokens(&message["content"]);
        }

        num_tokens += 3; // every reply is primed with <|start|>assistant<|message|> ??? TODO: check this
        num_tokens
    }
}

#[derive(Clone)]
pub enum ApiLlmType {
    OpenAi,
    Anthropic,
}
