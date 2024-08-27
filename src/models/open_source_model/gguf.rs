use super::*;
use chat_template::chat_template_from_local;
use gguf_rs::get_gguf_container;
use model_config_json::{model_config_json_from_gguf, model_config_json_from_local};
use std::path::Path;

/// Loads a GGUF model from Hugging Face or local path
/// Requires either hf_quant_file_url OR quant_file_path to be set.
/// Because there is currently no way to init a tokenizer from a GGUF in Rust,
/// Loading the tokenizer requires the tokenizer.json from either hf_config_repo_id or local_config_path.
#[derive(Default)]
pub struct LlmGgufLoader {
    pub hf_quant_file_url: Option<String>,
    pub hf_config_repo_id: Option<String>,
    pub quant_file_path: Option<PathBuf>,
    pub local_config_path: Option<PathBuf>,
}

impl LlmGgufLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_id(&self) -> String {
        if let Some(hf_quant_file_url) = &self.hf_quant_file_url {
            HuggingFaceLoader::model_id_from_url(hf_quant_file_url)
        } else if let Some(quant_file_path) = &self.quant_file_path {
            quant_file_path.to_string_lossy().to_string()
        } else {
            panic!("Either hf_quant_file_url or quant_file_path must be provided")
        }
    }

    pub fn local_model_path(&self, hf_loader: &HuggingFaceLoader) -> Result<PathBuf> {
        if let Some(hf_quant_file_url) = &self.hf_quant_file_url {
            let (_, repo_id, gguf_model_filename) =
                HuggingFaceLoader::parse_full_model_url(hf_quant_file_url);

            let local_model_filename = hf_loader.load_file(gguf_model_filename, repo_id)?;

            HuggingFaceLoader::canonicalize_local_path(local_model_filename)
        } else if let Some(local_quant_file_path) = &self.quant_file_path {
            Ok(local_quant_file_path.to_owned())
        } else {
            panic!("Either hf_quant_file_url or quant_file_path must be provided")
        }
    }

    pub fn tokenizer(&self, hf_loader: &HuggingFaceLoader) -> Result<Option<Arc<LlmTokenizer>>> {
        if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            let tokenizer_json_path = hf_loader.load_file("tokenizer.json", hf_config_repo_id)?;

            Ok(Some(Arc::new(LlmTokenizer::new_from_tokenizer_json(
                &tokenizer_json_path,
            )?)))
        } else if let Some(local_config_path) = &self.local_config_path {
            Ok(Some(Arc::new(LlmTokenizer::new_from_tokenizer_json(
                &PathBuf::from(local_config_path).join("tokenizer.json"),
            )?)))
        } else {
            Ok(None)
        }
    }

    pub fn model_config_json(&self, hf_loader: &HuggingFaceLoader) -> Result<OsLlmConfigJson> {
        if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            let config_json_path = hf_loader.load_file("config.json", hf_config_repo_id)?;
            model_config_json_from_local(&config_json_path)
        } else if let Some(local_config_path) = &self.local_config_path {
            model_config_json_from_local(&PathBuf::from(local_config_path).join("config.json"))
        } else {
            model_config_json_from_gguf(self.local_model_path(hf_loader)?)
        }
    }

    pub fn chat_template(&self, hf_loader: &HuggingFaceLoader) -> Result<OsLlmChatTemplate> {
        let chat_template = if let Some(hf_config_repo_id) = &self.hf_config_repo_id {
            let tokenizer_json_path: PathBuf =
                hf_loader.load_file("tokenizer_config.json", hf_config_repo_id)?;
            chat_template_from_local(&tokenizer_json_path)
        } else if let Some(local_config_path) = &self.local_config_path {
            chat_template_from_local(
                &PathBuf::from(local_config_path).join("tokenizer_config.json"),
            )
        } else {
            let gguf_model =
                get_gguf_container(&self.local_model_path(hf_loader)?.to_string_lossy())?
                    .decode()?;
            let metadata = gguf_model.metadata();
            Ok(OsLlmChatTemplate {
                chat_template: metadata
                    .get("tokenizer.chat_template")
                    .ok_or_else(|| anyhow!("tokenizer.chat_template not found in metadata"))?
                    .as_str()
                    .ok_or_else(|| anyhow!("tokenizer.chat_template is not a valid string"))?
                    .to_string(),
                chat_template_path: None,
                bos_token: metadata
                    .get("tokenizer.ggml.bos_token_id")
                    .and_then(|value| value.as_u64())
                    .and_then(|id| id.try_into().ok())
                    .map(|id: u32| id.to_string()),
                eos_token: metadata
                    .get("tokenizer.ggml.eos_token_id")
                    .and_then(|value| value.as_u64())
                    .and_then(|id| id.try_into().ok())
                    .map(|id: u32| id.to_string()),
                unk_token: None,
                base_generation_prefix: None,
            })
        };
        chat_template
    }
}

pub trait LlmGgufTrait {
    fn gguf_loader(&mut self) -> &mut LlmGgufLoader;

    /// Sets the Hugging Face url to the quantized model file.
    /// The full url to the model on hugging face like:
    /// 'https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf'
    fn hf_quant_file_url<S: Into<String>>(&mut self, hf_quant_file_url: S) -> &mut Self {
        self.gguf_loader().hf_quant_file_url = Some(hf_quant_file_url.into());
        self
    }

    /// Sets the Hugging Face repo id for the model config. This is used for loading the tokenizer.
    fn hf_config_repo_id<S: Into<String>>(&mut self, hf_config_repo_id: S) -> &mut Self {
        self.gguf_loader().hf_config_repo_id = Some(hf_config_repo_id.into());
        self
    }

    /// Sets the local path to the quantized model file.
    /// Use the /full/path/and/filename.gguf
    fn local_quant_file_path<P: AsRef<Path>>(&mut self, quant_file_path: P) -> &mut Self {
        self.gguf_loader().quant_file_path = Some(quant_file_path.as_ref().to_owned());
        self
    }

    /// Sets the local path to the model config files. This is used for loading the tokenizer.
    /// The path should contain the tokenizer.json and tokenizer_config.json files.
    fn local_config_path<P: AsRef<Path>>(&mut self, local_config_path: P) -> &mut Self {
        self.gguf_loader().local_config_path = Some(local_config_path.as_ref().to_owned());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_hf_basic() -> Result<()> {
        let model = OsLlmLoader::default()
        .hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
            .load()
            ?;

        println!("{:#?}", model);
        Ok(())
    }

    #[test]
    fn load_hf_with_config() -> Result<()> {
        let model = OsLlmLoader::default()
        .hf_quant_file_url("https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf")
        .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
            .load()
            ?;

        println!("{:#?}", model);
        assert!(model.tokenizer.is_some());
        Ok(())
    }

    #[test]
    fn load_local_basic() -> Result<()> {
        let model = OsLlmLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267")
            .load()
            ?;

        println!("{:#?}", model);
        Ok(())
    }

    #[test]
    fn load_local_with_config() -> Result<()> {
        let model = OsLlmLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267")
        .local_config_path("/workspaces/test/llm_utils/src/models/open_source_model/preset/llama/llama3_8b_instruct")
            .load()
            ?;
        assert!(model.tokenizer.is_some());
        println!("{:#?}", model);
        Ok(())
    }
}
