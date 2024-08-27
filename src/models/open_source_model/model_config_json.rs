use anyhow::{anyhow, Result};
use gguf_rs::get_gguf_container;
use serde::Deserialize;
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

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

pub fn model_config_json_from_local(config_json_path: &PathBuf) -> Result<OsLlmConfigJson> {
    let file = File::open(config_json_path)?;
    let reader = BufReader::new(file);
    let config: OsLlmConfigJson = serde_json::from_reader(reader)?;
    Ok(config)
}

pub fn model_config_json_from_gguf<P: AsRef<Path>>(local_model_path: P) -> Result<OsLlmConfigJson> {
    let gguf_model = get_gguf_container(&local_model_path.as_ref().to_string_lossy())?.decode()?;
    let metadata: &std::collections::BTreeMap<String, serde_json::Value> = gguf_model.metadata();
    let model_type = metadata
        .get("general.architecture")
        .ok_or_else(|| anyhow!("general.architecture not found"))?
        .as_str()
        .ok_or_else(|| anyhow!("general.architecture is not a valid string"))?;

    Ok(OsLlmConfigJson {
        architectures: vec![metadata
            .get("general.architecture")
            .ok_or_else(|| anyhow!("general.architecture not found in metadata"))?
            .to_string()],
        attention_dropout: 0.0,
        bos_token_id: metadata
            .get("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| anyhow!("tokenizer.ggml.bos_token_id not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("tokenizer.ggml.bos_token_id is not a valid u64"))?
            .try_into()
            .unwrap(),
        eos_token_id: metadata
            .get("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| anyhow!("tokenizer.ggml.eos_token_id not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("tokenizer.ggml.eos_token_id is not a valid u64"))?
            .try_into()
            .unwrap(),
        hidden_act: "".to_string(),
        hidden_size: metadata
            .get(&format!("{model_type}.embedding_length"))
            .ok_or_else(|| anyhow!("{model_type}.embedding_length not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.embedding_length is not a valid u64"))?
            .try_into()
            .unwrap(),
        initializer_range: 0.2,
        intermediate_size: metadata
            .get(&format!("{model_type}.feed_forward_length"))
            .ok_or_else(|| anyhow!("{model_type}.feed_forward_length not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.feed_forward_length is not a valid u64"))?
            .try_into()
            .unwrap(),
        max_position_embeddings: metadata
            .get(&format!("{model_type}.context_length"))
            .ok_or_else(|| anyhow!("{model_type}.context_length not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.context_length is not a valid u64"))?
            .try_into()
            .unwrap(),
        model_type: model_type.to_string(),
        num_attention_heads: metadata
            .get(&format!("{model_type}.attention.head_count"))
            .ok_or_else(|| anyhow!("{model_type}.attention.head_count not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.attention.head_count is not a valid u64"))?
            .try_into()
            .unwrap(),
        num_hidden_layers: metadata
            .get(&format!("{model_type}.block_count"))
            .ok_or_else(|| anyhow!("{model_type}.block_count not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.block_count is not a valid u64"))?
            .try_into()
            .unwrap(),
        num_key_value_heads: metadata
            .get(&format!("{model_type}.attention.head_count_kv"))
            .ok_or_else(|| anyhow!("{model_type}.attention.head_count_kv"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.attention.head_count_kv is not a valid u64"))?
            .try_into()
            .unwrap(),
        rms_norm_eps: metadata
            .get(&format!("{model_type}.attention.layer_norm_rms_epsilon"))
            .ok_or_else(|| anyhow!("{model_type}.attention.layer_norm_rms_epsilon"))?
            .as_f64()
            .unwrap(),
        rope_theta: metadata
            .get(&format!("{model_type}.rope.freq_base"))
            .ok_or_else(|| anyhow!("{model_type}.rope.freq_base not found in metadata"))?
            .as_f64()
            .unwrap(),
        torch_dtype: "float16".to_string(),
        transformers_version: "".to_string(),
        use_cache: true,
        vocab_size: metadata
            .get(&format!("{model_type}.vocab_size"))
            .ok_or_else(|| anyhow!("{model_type}.vocab_size not found in metadata"))?
            .as_u64()
            .ok_or_else(|| anyhow!("{model_type}.vocab_size is not a valid u64"))?
            .try_into()
            .unwrap(),
    })
}
