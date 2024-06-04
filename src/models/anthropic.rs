use crate::tokenizer::LlmTokenizer;
use std::collections::HashMap;

pub struct AnthropicLlm {
    pub model_id: String,
    pub context_length: u32,
    pub cost_per_m_in_tokens: f32,
    pub max_tokens_output: u32,
    pub cost_per_m_out_tokens: f32,
    pub tokens_per_message: u16,
    pub tokenizer: Option<LlmTokenizer>,
}

impl AnthropicLlm {
    pub fn anthropic_model_from_model_id(model_id: &str) -> AnthropicLlm {
        if model_id.starts_with("claude-3-opus") {
            Self::claude_3_opus()
        } else if model_id.starts_with("claude-3-sonnet") {
            Self::claude_3_sonnet()
        } else if model_id.starts_with("claude-3-haiku") {
            Self::claude_3_haiku()
        } else {
            panic!("{model_id} not found");
        }
    }

    pub fn claude_3_opus() -> AnthropicLlm {
        AnthropicLlm {
            model_id: "claude-3-opus-20240229".to_string(),
            context_length: 200000,
            cost_per_m_in_tokens: 15.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 75.00,
            tokens_per_message: 3,
            tokenizer: None,
        }
    }

    pub fn claude_3_sonnet() -> AnthropicLlm {
        AnthropicLlm {
            model_id: "claude-3-sonnet-20240229".to_string(),
            context_length: 200000,
            cost_per_m_in_tokens: 3.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokenizer: None,
        }
    }

    pub fn claude_3_haiku() -> AnthropicLlm {
        AnthropicLlm {
            model_id: "claude-3-haiku-20240307".to_string(),
            context_length: 200000,
            cost_per_m_in_tokens: 0.75,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 1.25,
            tokens_per_message: 3,
            tokenizer: None,
        }
    }

    pub fn with_tokenizer(&mut self) {
        eprintln!("Anthropic does not have a publically available tokenizer. See this for more information: https://github.com/javirandor/anthropic-tokenizer");
        eprintln!("However, since Anthropic does not support logit bias, we don't have a use for an actual tokenizer. So we can use TikToken to count tokens.");
        self.tokenizer = Some(LlmTokenizer::new_tiktoken("gpt-4"));
    }

    pub fn anthropic_token_count_of_prompt(
        &self,
        tokenizer: &LlmTokenizer,
        anthropic_prompt: &HashMap<String, HashMap<String, String>>,
    ) -> u32 {
        let mut num_tokens: u32 = 0;
        for message in anthropic_prompt.values() {
            num_tokens += self.tokens_per_message as u32;
            num_tokens += tokenizer.count_tokens(&message["content"]);
        }

        num_tokens += 3; // every reply is primed with <|start|>assistant<|message|> ??? TODO: check this
        num_tokens
    }
}
