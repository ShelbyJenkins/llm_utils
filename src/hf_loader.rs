use anyhow::{anyhow, Result};
use dotenv::dotenv;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// See llm_utils/src/bin/model_loader_cli.rs for cli instructions
// Downloads to Path: "/root/.cache/huggingface/hub/

#[derive(Debug, Default)]
pub struct HuggingFaceLoader {
    pub hf_token: Option<String>,
    pub repo_id: Option<String>,
}

impl HuggingFaceLoader {
    pub fn new(hf_token: Option<String>) -> Self {
        Self {
            hf_token,
            repo_id: None,
        }
    }

    pub fn model_from_owner_and_repo(mut self, owner: &str, repo: &str) -> Self {
        self.repo_id = Some(format!("{}/{}", owner, repo));
        self
    }

    pub fn model_from_repo_id(mut self, repo_id: &str) -> Self {
        self.repo_id = Some(repo_id.to_string());
        self
    }

    pub fn hf_token(mut self, hf_token: &str) -> Self {
        self.hf_token = Some(hf_token.to_string());
        self
    }

    fn get_hf_token(&self) -> Option<String> {
        if let Some(hf_token) = &self.hf_token {
            Some(hf_token.to_owned())
        } else {
            dotenv().ok(); // Load .env file
            if let Ok(hf_token) = dotenv::var("HUGGING_FACE_TOKEN") {
                Some(hf_token)
            } else {
                None
            }
        }
    }
    pub async fn load_file(&self, file_name: &str) -> Result<PathBuf> {
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(true)
            .with_token(self.get_hf_token())
            .build()
            .unwrap();
        api.model(self.repo_id.clone().unwrap())
            .get(file_name)
            .await
            .map_err(|e| anyhow!(e))
    }

    pub async fn load_tokenizer(&self) -> Result<Tokenizer> {
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_progress(true)
            .with_token(self.get_hf_token())
            .build()
            .unwrap();
        let tokenizer_filename = api
            .model(self.repo_id.clone().unwrap())
            .get("tokenizer.json")
            .await
            .map_err(|e| anyhow!(e))?;

        Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow!(e))
    }

    pub fn canonicalize_local_path(local_path: PathBuf) -> Result<String> {
        Ok(local_path
            .canonicalize()
            .map_err(|e| anyhow!(e))?
            .display()
            .to_string())
    }

    pub fn parse_full_model_url(model_url: &str) -> (String, String, String) {
        if !model_url.starts_with("https://huggingface.co") {
            panic!("URL does not start with https://huggingface.co\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
        } else if !model_url.ends_with(".gguf") {
            panic!("URL does not end with .gguf\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
        } else {
            let parts: Vec<&str> = model_url.split('/').collect();
            if parts.len() < 5 {
                panic!("URL does not have enough parts\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf");
            }
            let model_id = parts[4].to_string();
            let repo_id = format!("{}/{}", parts[3], parts[4]);
            let gguf_model_filename = parts.last().unwrap_or(&"").to_string();
            (model_id, repo_id, gguf_model_filename)
        }
    }

    pub fn model_url_from_repo_and_local_filename(
        repo_id: &str,
        local_model_filename: &str,
    ) -> String {
        let filename = std::path::Path::new(local_model_filename)
            .file_name()
            .and_then(|os_str| os_str.to_str())
            .unwrap_or(local_model_filename);

        format!("https://huggingface.co/{}/blob/main/{}", repo_id, filename)
    }

    pub fn model_id_from_url(model_url: &str) -> String {
        let parts = Self::parse_full_model_url(model_url);
        parts.0
    }
}
