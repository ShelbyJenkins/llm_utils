use super::gguf::gguf_metadata::GgufMetadata;
use serde::Deserialize;
use std::{fs::File, io::BufReader};

#[derive(Debug, Deserialize, Clone)]
pub struct LocalLlmMetadata {
    pub attention_dropout: f64,
    pub hidden_size: u64, // embedding_length
    pub initializer_range: f64,
    pub intermediate_size: u64,       // feed_forward_length
    pub max_position_embeddings: u64, // context_length
    pub num_attention_heads: u64,     // head_count
    pub num_hidden_layers: u64,
    pub num_key_value_heads: u64,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub torch_dtype: String,
    pub vocab_size: u32,
    pub architectures: Vec<String>,
    #[serde(alias = "model_type")]
    pub architecture: String,
}

pub fn model_metadata_from_local(
    config_json_path: &std::path::PathBuf,
) -> crate::Result<LocalLlmMetadata> {
    let file = File::open(config_json_path)?;
    let reader = BufReader::new(file);
    let config: LocalLlmMetadata = serde_json::from_reader(reader)?;
    Ok(config)
}

pub fn model_metadata_from_gguf(metadata: &GgufMetadata) -> crate::Result<LocalLlmMetadata> {
    let architecture = metadata.architecture.clone();

    Ok(LocalLlmMetadata {
        attention_dropout: 0.0,
        initializer_range: 0.2,
        torch_dtype: "float16".to_string(),
        vocab_size: metadata.get_value(&[&architecture], "vocab_size")?,
        num_hidden_layers: metadata.get_value(&[&architecture], "block_count")?,
        hidden_size: metadata.get_value(&[&architecture], "embedding_length")?,
        intermediate_size: metadata.get_value(&[&architecture], "feed_forward_length")?,
        max_position_embeddings: metadata.get_value(&[&architecture], "context_length")?,
        num_attention_heads: metadata.get_value(&[&architecture, "attention"], "head_count")?,
        num_key_value_heads: metadata.get_value(&[&architecture, "attention"], "head_count_kv")?,
        rms_norm_eps: metadata
            .get_value(&[&architecture, "attention"], "layer_norm_rms_epsilon")?,
        rope_theta: metadata.get_value(&[&architecture, "rope"], "freq_base")?,
        architectures: vec![architecture.clone()],
        architecture,
    })
}
