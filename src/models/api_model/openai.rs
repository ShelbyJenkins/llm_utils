use super::{ApiLlm, ApiLlmType};
use crate::tokenizer::LlmTokenizer;
use std::sync::Arc;

impl ApiLlm {
    pub fn openai_model_from_model_id(model_id: &str) -> ApiLlm {
        match model_id {
            "gpt-4" => Self::gpt_4(),
            "gpt-4-32k" => Self::gpt_4_32k(),
            "gpt-4-turbo" => Self::gpt_4_turbo(),
            "gpt-4o" => Self::gpt_4_o(),
            "gpt-3.5-turbo" => Self::gpt_3_5_turbo(),
            "gpt-4o-mini" => Self::gpt_3_5_turbo(),
            _ => panic!("Model ID ({model_id}) not found for ApiLlm"),
        }
    }

    pub fn gpt_4() -> ApiLlm {
        let model_id = "gpt-4".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 8192,
            cost_per_m_in_tokens: 30.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 60.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }

    pub fn gpt_4_32k() -> ApiLlm {
        let model_id = "gpt-4-32k".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 32768,
            cost_per_m_in_tokens: 60.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 120.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }

    pub fn gpt_4_turbo() -> ApiLlm {
        let model_id = "gpt-4-turbo".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 128000,
            cost_per_m_in_tokens: 10.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 30.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }

    pub fn gpt_4_o() -> ApiLlm {
        let model_id = "gpt-4o".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 128000,
            cost_per_m_in_tokens: 5.00,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 15.00,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }

    pub fn gpt_3_5_turbo() -> ApiLlm {
        let model_id = "gpt-3.5-turbo".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 16385,
            cost_per_m_in_tokens: 0.50,
            max_tokens_output: 4096,
            cost_per_m_out_tokens: 1.50,
            tokens_per_message: 4,
            tokens_per_name: Some(-1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }

    pub fn gpt_4_o_mini() -> ApiLlm {
        let model_id = "gpt-4o-mini".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLlm {
            model_id,
            context_length: 128000,
            cost_per_m_in_tokens: 0.15,
            max_tokens_output: 16384,
            cost_per_m_out_tokens: 0.60,
            tokens_per_message: 3,
            tokens_per_name: Some(1),
            tokenizer,
            api_type: ApiLlmType::OpenAi,
        }
    }
}

fn model_tokenizer(model_id: &str) -> Arc<LlmTokenizer> {
    Arc::new(
        LlmTokenizer::new_tiktoken(model_id)
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for {model_id}")),
    )
}

pub trait OpenAiModelTrait: Sized {
    fn model(&mut self) -> &mut Option<ApiLlm>;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::openai_model_from_model_id(model_id));
        self
    }

    /// Use gpt-4 as the model for the OpenAI client.
    fn gpt_4(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::gpt_4());
        self
    }

    /// Use gpt-4-32k as the model for the OpenAI client. Limited support for this model from OpenAI.
    fn gpt_4_32k(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::gpt_4_32k());
        self
    }

    /// Use gpt-4-turbo as the model for the OpenAI client.
    fn gpt_4_turbo(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::gpt_4_turbo());
        self
    }

    /// Use gpt-4-o as the model for the OpenAI client.
    fn gpt_4_o(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::gpt_4_o());
        self
    }

    /// Use gpt-3.5-turbo as the model for the OpenAI client.
    fn gpt_3_5_turbo(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = Some(ApiLlm::gpt_3_5_turbo());
        self
    }
}
