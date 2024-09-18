use crate::tokenizer::LlmTokenizer;

pub mod api_model;
pub mod local_model;

#[derive(Clone)]
pub struct LlmModelBase {
    pub model_id: String,
    pub context_length: u32,
    pub max_tokens_output: u32,
    pub tokenizer: std::sync::Arc<LlmTokenizer>,
}
