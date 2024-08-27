use super::{ApiLlm, ApiLlmType};
use crate::tokenizer::LlmTokenizer;
use std::sync::Arc;

impl ApiLlm {
    pub fn anthropic_model_from_model_id(model_id: &str) -> ApiLlm {
        if model_id.starts_with("claude-3-opus") {
            Self::claude_3_opus()
        } else if model_id.starts_with("claude-3-sonnet") {
            Self::claude_3_sonnet()
        } else if model_id.starts_with("claude-3-haiku") {
            Self::claude_3_haiku()
        } else if model_id.starts_with("claude-3.5-sonnet") {
            Self::claude_3_5_sonnet()
        } else {
            panic!("Model ID ({model_id}) not found for ApiLlm")
        }
    }

    pub fn claude_3_opus() -> ApiLlm {
        let model_id = "claude-3-opus-20240229".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 200000,
            cost_per_m_in_tokens: 15.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 75.00,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }

    pub fn claude_3_sonnet() -> ApiLlm {
        let model_id = "claude-3-sonnet-20240229".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 200000,
            cost_per_m_in_tokens: 3.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }

    pub fn claude_3_haiku() -> ApiLlm {
        let model_id = "claude-3-haiku-20240307".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 200000,
            cost_per_m_in_tokens: 0.75,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 1.25,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }

    pub fn claude_3_5_sonnet() -> ApiLlm {
        let model_id = "claude-3-5-sonnet-20240620".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 200000,
            cost_per_m_in_tokens: 3.00,
            max_tokens_output: 8192,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }
}

pub fn model_tokenizer(_model_id: &str) -> Arc<LlmTokenizer> {
    eprintln!("Anthropic does not have a publically available tokenizer. See this for more information: https://github.com/javirandor/anthropic-tokenizer");
    eprintln!("However, since Anthropic does not support logit bias, we don't have a use for an actual tokenizer. So we can use TikToken to count tokens.");
    Arc::new(
        LlmTokenizer::new_tiktoken("gpt-4")
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
    )
}

pub trait AnthropicModelTrait: Sized {
    fn model(&mut self) -> &mut Option<ApiLlm>;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::anthropic_model_from_model_id(model_id));
        self
    }

    /// Use the Claude 3 Opus model for the Anthropic client.
    fn claude_3_opus(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::claude_3_opus());
        self
    }

    /// Use the Claude 3 Sonnet model for the Anthropic client.
    fn claude_3_sonnet(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::claude_3_sonnet());
        self
    }

    /// Use the Claude 3 Haiku model for the Anthropic client.
    fn claude_3_haiku(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::claude_3_haiku());
        self
    }

    /// Use the Claude 3.5 Sonnet model for the Anthropic client.
    fn claude_3_5_sonnet(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::claude_3_5_sonnet());
        self
    }
}
