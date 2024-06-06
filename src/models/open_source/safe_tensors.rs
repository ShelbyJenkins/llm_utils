use super::*;

/// Not yet fully supported.
#[derive(Default)]
pub struct SafeTensorsModelBuilder {
    pub hf_token: Option<String>,
    pub repo_id: String,
    pub model_id: String,
    pub with_tokenizer: bool,
}

impl SafeTensorsModelBuilder {
    pub fn new(hf_token: Option<String>, repo_id: String) -> Self {
        if let Some((_, model_id)) = repo_id.clone().split_once('/') {
            Self {
                hf_token,
                repo_id,
                with_tokenizer: false,
                model_id: model_id.to_string(),
            }
        } else {
            panic!("Invalid repo_id format: {}", repo_id)
        }
    }

    pub fn with_tokenizer(mut self, with_tokenizer: bool) -> Self {
        self.with_tokenizer = with_tokenizer;
        self
    }

    pub fn load(&mut self) -> Result<OsLlm> {
        let hf_loader =
            HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(&self.repo_id);
        let config_json_path = hf_loader.load_file("config.json")?;
        let model_config_json = model_config_json_from_local(&config_json_path)?;

        let tokenizer_config_json_path = hf_loader.load_file("tokenizer_config.json")?;
        let chat_template = chat_template_from_local(&tokenizer_config_json_path)?;

        let tokenizer = if self.with_tokenizer {
            let tokenizer_json_path: PathBuf = hf_loader.load_file("tokenizer.json")?;
            Some(LlmTokenizer::new_from_tokenizer_json(&tokenizer_json_path))
        } else {
            None
        };

        Ok(OsLlm {
            model_id: self.model_id.clone(),
            model_url: HuggingFaceLoader::model_url_from_repo(&self.repo_id),
            model_config_json,
            chat_template,
            local_model_paths: hf_loader.load_model_safe_tensors()?,
            tokenizer,
        })
    }
}

pub fn model_config_json_from_local(config_json_path: &PathBuf) -> Result<OsLlmConfigJson> {
    let file = File::open(config_json_path)?;
    let reader = BufReader::new(file);
    let config: OsLlmConfigJson = serde_json::from_reader(reader)?;
    Ok(config)
}

pub fn chat_template_from_local(tokenizer_config_json_path: &PathBuf) -> Result<OsLlmChatTemplate> {
    let file = File::open(tokenizer_config_json_path)?;
    let reader = BufReader::new(file);
    let mut tokenizer_config: OsLlmChatTemplate = serde_json::from_reader(reader)?;
    tokenizer_config.chat_template_path =
        Some(tokenizer_config_json_path.to_string_lossy().to_string());
    Ok(tokenizer_config)
}

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct OsLlmConfigJson {
    pub architectures: Vec<String>,
    pub attention_dropout: f64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub hidden_act: String,
    pub hidden_size: i64, // embedding_length
    pub initializer_range: f64,
    pub intermediate_size: i64,       // feed_forward_length
    pub max_position_embeddings: i64, // context_length
    pub model_type: String,
    pub num_attention_heads: i64, // head_count
    pub num_hidden_layers: i64,
    pub num_key_value_heads: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i64,
}

// This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
pub fn estimate_context_size(
    model_config_json: &OsLlmConfigJson,
    ctx_size: i64,
    batch_size: i64,
) -> f64 {
    let input_buffer = input_buffer(model_config_json, ctx_size, batch_size);
    let compute_buffer = compute_buffer(model_config_json, ctx_size);
    let kv_cache = kv_cache(model_config_json, ctx_size);
    let context_bits = input_buffer + kv_cache + compute_buffer;
    context_bits / (1024f64 * 1024f64 * 1024f64)
}

fn input_buffer(model_config_json: &OsLlmConfigJson, ctx_size: i64, batch_size: i64) -> f64 {
    ((batch_size * 3)
        + (model_config_json.hidden_size * batch_size)
        + (batch_size * ctx_size)
        + ctx_size) as f64
}

fn compute_buffer(model_config_json: &OsLlmConfigJson, ctx_size: i64) -> f64 {
    (ctx_size as f64 / 1024f64 * 2f64 + 0.75)
        * model_config_json.num_attention_heads as f64
        * 1024f64
        * 1024f64
}

fn kv_cache(model_config_json: &OsLlmConfigJson, ctx_size: i64) -> f64 {
    let cache_bit = match model_config_json.torch_dtype.as_str() {
        "float32" => 32,
        "float16" | "bfloat16" => 16,
        _ => panic!("Unsupported data type"),
    };
    let n_gqa = model_config_json.num_attention_heads / model_config_json.num_key_value_heads;
    let n_embd_gqa = model_config_json.hidden_size / n_gqa;
    let n_elements = n_embd_gqa * (model_config_json.num_hidden_layers * ctx_size);
    let size = 2 * n_elements;
    size as f64 * (cache_bit as f64 / 8f64)
}
