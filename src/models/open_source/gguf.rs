use super::*;
use gguf_rs::get_gguf_container;

/// Loads a GGUF model from Hugging Face or local path
/// Requires either hf_quant_file_url OR quant_file_path to be set.
/// Because there is currently no way to init a tokenizer from a GGUF in Rust,
/// Loading the tokenizer requires the tokenizer.json from either hf_config_repo_id or local_config_path.
#[derive(Default)]
pub struct GGUFModelBuilder {
    pub hf_token: Option<String>,
    pub hf_quant_file_url: Option<String>,
    pub hf_config_repo_id: Option<String>,
    pub quant_file_path: Option<String>,
    pub local_config_path: Option<String>,
}

impl GGUFModelBuilder {
    pub fn new() -> Self {
        Self {
            hf_token: None,
            hf_quant_file_url: None,
            hf_config_repo_id: None,
            quant_file_path: None,
            local_config_path: None,
        }
    }

    /// Sets the Hugging Face token to use for private models.
    pub fn hf_token(&mut self, hf_token: &str) -> &mut Self {
        self.hf_token = Some(hf_token.to_owned());
        self
    }

    /// Sets the Hugging Face url to the quantized model file.
    /// The full url to the model on hugging face like:
    /// 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf'
    pub fn hf_quant_file_url(&mut self, hf_quant_file_url: &str) -> &mut Self {
        self.hf_quant_file_url = Some(hf_quant_file_url.to_owned());
        self
    }

    /// Sets the Hugging Face repo id for the model config. This is used for loading the tokenizer.
    pub fn hf_config_repo_id(&mut self, hf_config_repo_id: &str) -> &mut Self {
        self.hf_config_repo_id = Some(hf_config_repo_id.to_owned());
        self
    }

    /// Sets the local path to the quantized model file.
    /// Use the /full/path/and/filename.gguf
    pub fn local_quant_file_path(&mut self, quant_file_path: &str) -> &mut Self {
        self.quant_file_path = Some(quant_file_path.to_owned());
        self
    }

    /// Sets the local path to the model config files. This is used for loading the tokenizer.
    /// The path should contain the tokenizer.json and tokenizer_config.json files.
    pub fn local_config_path(&mut self, local_config_path: &str) -> &mut Self {
        self.local_config_path = Some(local_config_path.to_owned());
        self
    }

    pub fn load(&mut self) -> Result<OsLlm> {
        let local_model_path = self.load_model()?;

        let (model_url, model_id) = if let Some(hf_quant_file_url) = &self.hf_quant_file_url {
            (
                hf_quant_file_url.to_owned(),
                HuggingFaceLoader::model_id_from_url(hf_quant_file_url),
            )
        } else {
            (local_model_path.to_string(), local_model_path.to_string())
        };

        let gguf_model = get_gguf_container(&local_model_path)?.decode()?;
        let metadata = gguf_model.metadata();

        Ok(OsLlm {
            model_id,
            model_url,
            model_config_json: self.load_config(metadata)?,
            chat_template: self.load_chat_template(metadata)?,
            local_model_paths: vec![local_model_path],
            tokenizer: self.load_tokenizer()?,
        })
    }

    fn load_model(&self) -> Result<String> {
        if let Some(hf_quant_file_url) = &self.hf_quant_file_url {
            let (_, repo_id, gguf_model_filename) =
                HuggingFaceLoader::parse_full_model_url(hf_quant_file_url);
            let hf_loader =
                Some(HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(&repo_id));

            let local_model_filename = hf_loader
                .as_ref()
                .unwrap()
                .load_file(&gguf_model_filename)?;
            HuggingFaceLoader::canonicalize_local_path(local_model_filename)
        } else if let Some(local_quant_file_path) = &self.quant_file_path {
            Ok(local_quant_file_path.to_owned())
        } else {
            panic!("Either hf_quant_file_url or quant_file_path must be provided")
        }
    }

    fn load_tokenizer(&self) -> Result<Option<LlmTokenizer>> {
        if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            Ok(Some(LlmTokenizer::new_from_hf_repo(
                &self.hf_token,
                hf_config_repo_id,
            )?))
        } else if let Some(local_config_path) = &self.local_config_path {
            Ok(Some(LlmTokenizer::new_from_tokenizer_json(
                &PathBuf::from(local_config_path).join("tokenizer.json"),
            )))
        } else {
            Ok(None)
        }
    }

    fn load_config(
        &self,
        metadata: &std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<OsLlmConfigJson> {
        if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            let hf_loader =
                HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(hf_config_repo_id);
            let config_json_path = hf_loader.load_file("config.json")?;
            model_config_json_from_local(&config_json_path)
        } else if let Some(local_config_path) = &self.local_config_path {
            model_config_json_from_local(&PathBuf::from(local_config_path).join("config.json"))
        } else {
            model_config_json_from_gguf(metadata)
        }
    }

    fn load_chat_template(
        &self,
        metadata: &std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<OsLlmChatTemplate> {
        let chat_template = if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            let hf_loader =
                HuggingFaceLoader::new(self.hf_token.clone()).model_from_repo_id(hf_config_repo_id);
            let tokenizer_json_path: PathBuf = hf_loader.load_file("tokenizer_config.json")?;
            chat_template_from_local(&tokenizer_json_path)
        } else if let Some(local_config_path) = &self.local_config_path {
            chat_template_from_local(
                &PathBuf::from(local_config_path).join("tokenizer_config.json"),
            )
        } else {
            Err(anyhow!("_"))
        };
        if chat_template.is_ok() {
            chat_template
        } else {
            Ok(OsLlmChatTemplate {
                chat_template: metadata
                    .get("tokenizer.chat_template")
                    .ok_or_else(|| anyhow!("tokenizer.chat_template not found in metadata"))?
                    .as_str()
                    .ok_or_else(|| anyhow!("tokenizer.chat_template is not a valid string"))?
                    .to_string(),
                chat_template_path: None,
                bos_token: None,
                eos_token: None,
                unk_token: None,
            })
        }
    }
}

fn model_config_json_from_gguf(
    metadata: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<OsLlmConfigJson> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_hf_basic() -> Result<()> {
        let model = GGUFModelBuilder::new().hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
            .load()
            ?;

        println!("{:#?}", model);
        Ok(())
    }

    #[test]
    fn load_hf_with_config() -> Result<()> {
        let model = GGUFModelBuilder::new().hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
            .load()
            ?;

        println!("{:#?}", model);
        assert!(model.tokenizer.is_some());
        Ok(())
    }

    #[test]
    fn load_local_basic() -> Result<()> {
        let model = GGUFModelBuilder::new().local_quant_file_path("/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267")
            .load()
            ?;

        println!("{:#?}", model);
        Ok(())
    }

    #[test]
    fn load_local_with_config() -> Result<()> {
        let model = GGUFModelBuilder::new().local_quant_file_path("/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267").local_config_path("/workspaces/test/llm_utils/src/models/open_source/preset/llama/llama_3_8b_instruct")
            .load()
            ?;
        assert!(model.tokenizer.is_some());
        println!("{:#?}", model);
        Ok(())
    }
}
