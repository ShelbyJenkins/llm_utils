pub mod preset_loader;

use super::*;
use chat_template::chat_template_from_local;
use model_config_json::model_config_json_from_local;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::LazyLock,
};
use vram::{estimate_context_size, quantization_from_vram};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlmPresetData {
    pub model_id: String,
    pub gguf_repo_id: String,
    pub number_of_parameters: u64,
    pub f_name_for_q_bits: QuantizationConfig,
    pub base_generation_prefix: String,
}

impl LlmPresetData {
    pub fn new<P: AsRef<Path>>(path: P) -> LlmPresetData {
        let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = PathBuf::from(cargo_manifest_dir)
            .join("src")
            .join("models")
            .join("open_source_model")
            .join("preset")
            .join(path)
            .join("model_macro_data.json");
        let mut file = File::open(&path)
            .unwrap_or_else(|_| panic!("Failed to open file at {}", path.display()));
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read file");
        serde_json::from_str(&contents).expect("Failed to parse JSON")
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizationConfig {
    pub q8: Option<String>,
    pub q7: Option<String>,
    pub q6: Option<String>,
    pub q5: Option<String>,
    pub q4: Option<String>,
    pub q3: Option<String>,
    pub q2: Option<String>,
    pub q1: Option<String>,
}

macro_rules! generate_models {
    ($enum_name:ident {
        $($variant:ident => $path:expr),* $(,)?
    }) => {
        #[derive(Debug, Clone)]
        pub enum $enum_name {
            $($variant),*
        }

        impl $enum_name {
            fn get_data(&self) -> &'static LlmPresetData {
                match self {
                    $(
                        Self::$variant => {
                            static DATA: LazyLock<LlmPresetData> = LazyLock::new(|| {
                                LlmPresetData::new($path)
                            });
                            &DATA
                        }
                    ),*
                }
            }

            pub fn model_id(&self) -> String {
                self.get_data().model_id.to_string()
            }

            pub fn gguf_repo_id(&self) -> &str {
                &self.get_data().gguf_repo_id
            }

            pub fn model_config_json(&self) -> Result<OsLlmConfigJson> {
                let preset_config_path = self.preset_config_path();
                let config_json_path = preset_config_path.join("config.json");
                model_config_json_from_local(&config_json_path)
            }

            pub fn chat_template(&self) -> Result<OsLlmChatTemplate> {
                let preset_config_path = self.preset_config_path();
                let tokenizer_config_json_path = preset_config_path.join("tokenizer_config.json");
                let mut chat_template = chat_template_from_local(&tokenizer_config_json_path)?;
                chat_template.base_generation_prefix = Some(self.get_data().base_generation_prefix.clone());
                Ok(chat_template)
            }

            pub fn tokenizer(&self) -> Result<Arc<LlmTokenizer>> {
                let preset_config_path = self.preset_config_path();
                let tokenizer_json_path = preset_config_path.join("tokenizer.json");
                Ok(Arc::new(LlmTokenizer::new_from_tokenizer_json(
                    &tokenizer_json_path,
                )?))
            }

            pub fn f_name_for_q_bits(&self, q_bits: u8) -> Option<String> {
                match q_bits {
                    8 => self.get_data().f_name_for_q_bits.q8.clone(),
                    7 => self.get_data().f_name_for_q_bits.q7.clone(),
                    6 => self.get_data().f_name_for_q_bits.q6.clone(),
                    5 => self.get_data().f_name_for_q_bits.q5.clone(),
                    4 => self.get_data().f_name_for_q_bits.q4.clone(),
                    3 => self.get_data().f_name_for_q_bits.q3.clone(),
                    2 => self.get_data().f_name_for_q_bits.q2.clone(),
                    1 => self.get_data().f_name_for_q_bits.q1.clone(),
                    _ => panic!("Quantization bits must be between 1 and 8"),
                }

            }

            pub fn number_of_parameters(&self) -> f64 {
                self.get_data().number_of_parameters as f64 * 1_000_000_000.0
            }

            pub fn preset_config_path(&self) -> PathBuf {
                match self {
                    $(
                        Self::$variant => {
                            let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                            PathBuf::from(cargo_manifest_dir)
                                .join("src")
                                .join("models")
                                .join("open_source_model")
                                .join("preset")
                                .join($path)
                        }
                    ),*
                }

            }

        }

        pub trait LlmPresetTrait {
            fn preset_loader(&mut self) -> &mut LlmPresetLoader;

            fn available_vram(mut self, available_vram: u32) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().available_vram = available_vram;
                self
            }

            fn use_ctx_size(mut self, use_ctx_size: u32) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().use_ctx_size = Some(use_ctx_size);
                self
            }

            $(
                paste::paste! {
                    fn [<$variant:snake>](mut self) -> Self
                    where
                    Self: Sized,
                    {
                        self.preset_loader().llm_preset = $enum_name::$variant;
                        self
                    }
                }
            )*
        }
    };
}

generate_models!(
    LlmPreset {
        Llama3_1_8bInstruct => "llama/llama3_1_8b_instruct",
        Llama3_8bInstruct => "llama/llama3_8b_instruct",
        Llama3_70bInstruct => "llama/llama3_70b_instruct",
        Mistral7bInstructV0_3 => "mistral/mistral7b_instruct_v0_3",
        Mixtral8x7bInstructV0_1 => "mistral/mixtral8x7b_instruct_v0_1",
        MistralNemoInstruct2407 => "mistral/mistral_nemo_instruct_2407",
        Phi3Medium4kInstruct => "phi/phi3_medium4k_instruct",
        Phi3Mini4kInstruct => "phi/phi3_mini4k_instruct",
        Phi3_5MiniInstruct => "phi/phi3_5_mini_instruct",
    }
);

#[allow(clippy::derivable_impls)]
impl Default for LlmPreset {
    fn default() -> Self {
        LlmPreset::Llama3_8bInstruct
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_macros_test() -> Result<()> {
        let variants = vec![
            LlmPreset::Llama3_1_8bInstruct,
            LlmPreset::Llama3_8bInstruct,
            LlmPreset::Llama3_70bInstruct,
            LlmPreset::Mistral7bInstructV0_3,
            LlmPreset::Mixtral8x7bInstructV0_1,
            LlmPreset::MistralNemoInstruct2407,
            LlmPreset::Phi3Medium4kInstruct,
            LlmPreset::Phi3Mini4kInstruct,
            LlmPreset::Phi3_5MiniInstruct,
        ];
        for variant in variants {
            println!("{:#?}", variant.model_id());
            println!("{:#?}", variant.gguf_repo_id());
            println!("{:#?}", variant.model_config_json());
            println!("{:#?}", variant.chat_template());
            println!("{:#?}", variant.tokenizer());
            println!("{:#?}", variant.number_of_parameters());
            println!("{:#?}", variant.preset_config_path());
            for i in 1..=8 {
                println!("{:#?}", variant.f_name_for_q_bits(i));
            }
        }
        Ok(())
    }

    #[test]
    fn load() -> Result<()> {
        let model: OsLlm = OsLlmLoader::default()
            .llama3_1_8b_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model: OsLlm = OsLlmLoader::default()
            .llama3_8b_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .llama3_70b_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .mistral7b_instruct_v0_3()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .mixtral8x7b_instruct_v0_1()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .mistral_nemo_instruct2407()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .phi3_medium4k_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model = OsLlmLoader::default()
            .phi3_mini4k_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        let model: OsLlm = OsLlmLoader::default()
            .phi3_5_mini_instruct()
            .available_vram(48)
            .load()?;

        println!("{:#?}", model);

        Ok(())
    }
}
