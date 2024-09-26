use crate::tokenizer::LlmTokenizer;

pub mod api_model;
pub mod local_model;

#[derive(Clone)]
pub struct LlmModelBase {
    pub model_id: String,
    pub model_ctx_size: u64,
    pub inference_ctx_size: u64,
    pub tokenizer: std::sync::Arc<LlmTokenizer>,
}
