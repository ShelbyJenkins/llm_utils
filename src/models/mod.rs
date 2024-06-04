pub mod anthropic;
pub mod open_source;
pub mod openai;

pub use open_source::{
    gguf::GGUFModelBuilder,
    preset::{LlmPreset, PresetModelBuilder},
    safe_tensors::SafeTensorsModelBuilder,
    OsLlm,
};
