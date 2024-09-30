use crate::models::{
    local_model::{
        gguf::{load_chat_template, load_tokenizer},
        metadata::LocalLlmMetadata,
        LocalLlmModel,
    },
    LlmModelBase,
};

#[derive(Default, Clone)]
pub struct GgufLocalLoader {
    pub local_quant_file_path: Option<std::path::PathBuf>,
    pub local_config_path: Option<std::path::PathBuf>,
    pub local_tokenizer_path: Option<std::path::PathBuf>,
    pub local_tokenizer_config_path: Option<std::path::PathBuf>,
    pub model_id: Option<String>,
}

impl GgufLocalLoader {
    pub fn load(&mut self) -> crate::Result<LocalLlmModel> {
        let local_model_path =
            if let Some(local_quant_file_path) = self.local_quant_file_path.as_ref() {
                local_quant_file_path.to_owned()
            } else {
                crate::bail!("local_quant_file_path must be set")
            };

        let model_id = if let Some(model_id) = &self.model_id {
            model_id.to_owned()
        } else {
            local_model_path.to_string_lossy().to_string()
        };

        let model_metadata = LocalLlmMetadata::from_gguf_path(&local_model_path)?;

        Ok(LocalLlmModel {
            model_base: LlmModelBase {
                model_id,
                model_ctx_size: model_metadata.context_length(),
                inference_ctx_size: model_metadata.context_length(),
                tokenizer: load_tokenizer(&self.local_tokenizer_path, &model_metadata)?,
            },
            chat_template: load_chat_template(&self.local_tokenizer_config_path, &model_metadata)?,
            model_metadata,
            local_model_path,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::models::local_model::{gguf::GgufLoader, GgufLoaderTrait};

    #[test]
    fn load_local_basic() {
        let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
            .load()
           .unwrap();

        println!("{:#?}", model);
    }

    #[test]
    fn load_local_with_config() {
        let model = GgufLoader::default()
        .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
        .local_config_path("/workspaces/test/llm_utils/src/models/local_model/gguf/preset/llama/llama3_1_8b_instruct/config.json")
            .load()
           .unwrap();

        println!("{:#?}", model);
    }
}
