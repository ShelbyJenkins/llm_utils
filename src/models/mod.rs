pub mod anthropic;
pub mod gguf;
pub mod openai;

#[derive(Clone, Default)]
pub enum OpenSourceModelType {
    #[default]
    Mistral7bInstructV0_3,
    Mixtral8x7bInstruct,
    Mixtral8x22bInstruct,
    Llama3_70bInstruct,
    Llama3_8bInstruct,
}

impl OpenSourceModelType {
    pub fn model_id(&self) -> String {
        match self {
            OpenSourceModelType::Mistral7bInstructV0_3 => "Mistral-7B-Instruct-v0.3".to_string(),
            OpenSourceModelType::Mixtral8x7bInstruct => "Mixtral-8x7B-Instruct-v0.1".to_string(),
            OpenSourceModelType::Mixtral8x22bInstruct => "Mixtral-8x22B-Instruct-v0.1".to_string(),
            OpenSourceModelType::Llama3_70bInstruct => "Meta-Llama-3-70B-Instruct".to_string(),
            OpenSourceModelType::Llama3_8bInstruct => "Meta-Llama-3-8B-Instruct".to_string(),
        }
    }
    pub fn safe_tensors_repo_id(&self) -> String {
        match self {
            OpenSourceModelType::Mistral7bInstructV0_3 => {
                "mistralai/Mistral-7B-Instruct-v0.3".to_string()
            }
            OpenSourceModelType::Mixtral8x7bInstruct => {
                "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string()
            }
            OpenSourceModelType::Mixtral8x22bInstruct => {
                "mistralai/Mixtral-8x22B-Instruct-v0.1".to_string()
            }
            OpenSourceModelType::Llama3_70bInstruct => {
                "meta-llama/Meta-Llama-3-70B-Instruct".to_string()
            }
            OpenSourceModelType::Llama3_8bInstruct => {
                "meta-llama/Meta-Llama-3-8B-Instruct".to_string()
            }
        }
    }

    pub fn gguf_repo_id(&self) -> String {
        match self {
            OpenSourceModelType::Mistral7bInstructV0_3 => {
                "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF".to_string()
            }
            OpenSourceModelType::Mixtral8x7bInstruct => {
                "MaziyarPanahi/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string()
            }
            OpenSourceModelType::Mixtral8x22bInstruct => {
                "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF".to_string()
            }
            OpenSourceModelType::Llama3_70bInstruct => {
                "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF".to_string()
            }
            OpenSourceModelType::Llama3_8bInstruct => {
                "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF".to_string()
            }
        }
    }

    pub fn number_of_parameters(&self) -> u32 {
        match self {
            OpenSourceModelType::Mistral7bInstructV0_3 => 7,
            OpenSourceModelType::Mixtral8x7bInstruct => 56,
            OpenSourceModelType::Mixtral8x22bInstruct => 176,
            OpenSourceModelType::Llama3_70bInstruct => 70,
            OpenSourceModelType::Llama3_8bInstruct => 8,
        }
    }
}

pub fn quantization_from_vram(parameters: u32, vram_gb: u32, input_output_max_reqs_gb: f64) -> u8 {
    let cuda_overhead_gb = 0.777;
    let memory_gb = vram_gb as f64 - input_output_max_reqs_gb - cuda_overhead_gb;
    let memory_bytes = memory_gb * 1024.0 * 1024.0 * 1024.0;
    let num_params = parameters as f64 * 1_000_000_000.0 * 1.1;

    match memory_bytes {
        x if x >= num_params * 1.0 => 8,
        x if x >= num_params * 0.875 => 7,
        x if x >= num_params * 0.75 => 6,
        x if x >= num_params * 0.625 => 5,
        x if x >= num_params * 0.5 => 4,
        x if x >= num_params * 0.375 => 3,
        x if x >= num_params * 0.25 => 2,
        x if x >= num_params * 0.125 => 1,
        _ => panic!("Not enough VRAM!"),
    }
}
