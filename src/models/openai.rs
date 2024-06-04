use crate::tokenizer::LlmTokenizer;
use std::collections::HashMap;

pub struct OpenAiLlm {
    pub model_id: String,
    pub context_length: u32,
    pub cost_per_m_in_tokens: f32,
    pub max_tokens_output: u32,
    pub cost_per_m_out_tokens: f32,
    pub tokens_per_message: u32,
    pub tokens_per_name: i32,
    pub tokenizer: Option<LlmTokenizer>,
}

impl OpenAiLlm {
    pub fn openai_backend_from_model_id(model_id: &str) -> OpenAiLlm {
        match model_id {
            "gpt-4" => Self::gpt_4(),
            "gpt-4-32k" => Self::gpt_4_32k(),
            "gpt-4-turbo" => Self::gpt_4_turbo(),
            "gpt-4o" => Self::gpt_4_o(),
            "gpt-3.5-turbo" => Self::gpt_3_5_turbo(),
            _ => panic!("{model_id} not found"),
        }
    }

    pub fn gpt_4() -> OpenAiLlm {
        OpenAiLlm {
            model_id: "gpt-4".to_string(),
            context_length: 8192,
            cost_per_m_in_tokens: 30.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 60.00,
            tokens_per_message: 3,
            tokens_per_name: 1,
            tokenizer: None,
        }
    }

    pub fn gpt_4_32k() -> OpenAiLlm {
        OpenAiLlm {
            model_id: "gpt-4-32k".to_string(),
            context_length: 32768,
            cost_per_m_in_tokens: 60.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 120.00,
            tokens_per_message: 3,
            tokens_per_name: 1,
            tokenizer: None,
        }
    }

    pub fn gpt_4_turbo() -> OpenAiLlm {
        OpenAiLlm {
            model_id: "gpt-4-turbo".to_string(),
            context_length: 128000,
            cost_per_m_in_tokens: 10.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 30.00,
            tokens_per_message: 3,
            tokens_per_name: 1,
            tokenizer: None,
        }
    }

    pub fn gpt_4_o() -> OpenAiLlm {
        OpenAiLlm {
            model_id: "gpt-4o".to_string(),
            context_length: 128000,
            cost_per_m_in_tokens: 5.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: 1,
            tokenizer: None,
        }
    }

    pub fn gpt_3_5_turbo() -> OpenAiLlm {
        OpenAiLlm {
            model_id: "gpt-3.5-turbo".to_string(),
            context_length: 16385,
            cost_per_m_in_tokens: 0.50,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 1.50,
            tokens_per_message: 4,
            tokens_per_name: -1,
            tokenizer: None,
        }
    }

    pub fn with_tokenizer(&mut self) {
        self.tokenizer = Some(LlmTokenizer::new_tiktoken(&self.model_id));
    }

    pub fn openai_token_count_of_prompt(
        &self,
        tokenizer: &LlmTokenizer,
        openai_prompt: &HashMap<String, HashMap<String, String>>,
    ) -> u32 {
        let mut num_tokens = 0;
        for message in openai_prompt.values() {
            num_tokens += self.tokens_per_message;
            num_tokens += tokenizer.count_tokens(&message["content"]);

            if message["name"].is_empty() {
                if self.tokens_per_name < 0 {
                    // Handles cases for certain models where name doesn't count towards token count
                    num_tokens -= self.tokens_per_name.unsigned_abs();
                } else {
                    num_tokens += self.tokens_per_name as u32;
                }
            }
        }
        num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
        num_tokens
    }
}
