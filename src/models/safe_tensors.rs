// Unused for now. Will implement once mistral rs supports dual GPUs.

async fn load_model_config_json(&mut self, owner_repo_id: &str) -> Result<ModelConfigJson> {
    let config_json_path = HuggingFaceLoader::new(self.hf_token.clone())
        .model_from_repo_id(owner_repo_id)
        .load_file("config.json")
        .await?;

    let file = File::open(config_json_path)?;
    let reader = BufReader::new(file);
    let config: ModelConfigJson = serde_json::from_reader(reader)?;
    self.model_config_json = Some(config.clone());
    Ok(config)
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfigJson {
    pub architectures: Vec<String>,
    pub attention_dropout: f64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub hidden_act: String,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub model_type: String,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub num_key_value_heads: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<i64>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i64,
}

fn calculate_context_length_max_memory_requirements(config: &ModelConfigJson) -> f64 {
    let hidden_size = config.hidden_size;
    let num_attention_heads = config.num_attention_heads;
    let intermediate_size = config.intermediate_size;
    let vocab_size = config.vocab_size;
    let max_position_embeddings = config.max_position_embeddings;

    let attention_params = hidden_size * (num_attention_heads * 3);
    let ffn_params = hidden_size * intermediate_size * 2;
    let embedding_params = vocab_size * hidden_size * 2;
    let position_embedding_params = max_position_embeddings * hidden_size;

    let total_params = attention_params + ffn_params + embedding_params + position_embedding_params;

    // let bytes_per_param = match config.torch_dtype.as_str() {
    //     "float32" => 4,
    //     "float16" | "bfloat16" => 2,
    //     _ => panic!("Unsupported data type"),
    // };

    let memory_requirements_bytes = total_params * 2;

    memory_requirements_bytes as f64 / (1024.0 * 1024.0)
}
