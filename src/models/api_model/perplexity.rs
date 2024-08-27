use super::{ApiLlm, ApiLlmType};
use crate::tokenizer::LlmTokenizer;
use std::sync::Arc;

impl ApiLlm {
    pub fn perplexity_model_from_model_id(model_id: &str) -> ApiLlm {
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

    pub fn sonar_small() -> ApiLlm {
        let model_id = "llama-3.1-sonar-small-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 127072,
            cost_per_m_in_tokens: 0.1,
            max_tokens_output: 8192,
            cost_per_m_out_tokens: 0.1,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }

    pub fn sonar_large() -> ApiLlm {
        let model_id = "llama-3.1-sonar-large-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 127072,
            cost_per_m_in_tokens: 0.5,
            max_tokens_output: 8192,
            cost_per_m_out_tokens: 0.5,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }

    pub fn sonar_huge() -> ApiLlm {
        let model_id = "llama-3.1-sonar-huge-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 127072,
            cost_per_m_in_tokens: 2.5,
            max_tokens_output: 8192,
            cost_per_m_out_tokens: 2.5,
            tokens_per_message: 3,
            tokens_per_name: None,
            tokenizer,
            api_type: ApiLlmType::Anthropic,
        }
    }
}

pub fn model_tokenizer(_model_id: &str) -> Arc<LlmTokenizer> {
    Arc::new(
        LlmTokenizer::new_tiktoken("gpt-4")
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
    )
}

pub trait PerplexityModelTrait: Sized {
    fn model(&mut self) -> &mut Option<ApiLlm>;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::anthropic_model_from_model_id(model_id));
        self
    }

    fn sonar_small(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::sonar_small());
        self
    }

    fn sonar_large(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::sonar_large());
        self
    }

    fn sonar_huge(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::sonar_huge());
        self
    }
}
