use gguf::GgufLoader;
use llm_utils::{models::local_model::*, prompting::LlmPrompt};

fn main() {
    // Using
    let _model = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
        .load()
        .unwrap();

    // By default we attempt to extract everything we need from the GGUF file.
    // If you need to specifiy the tokenizer or chat template to use, you can add a hf repo to load from.
    let model = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    .hf_config_repo_id("meta-llama/Meta-Llama-3-8B-Instruct")
        .load()
        .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.

    let prompt: LlmPrompt = LlmPrompt::new_chat_template_prompt(&model);

    prompt
        .add_user_message()
        .unwrap()
        .set_content("This is a test prompt.");

    // To feed into the inference engine
    let built_prompt = prompt.get_built_prompt_string().unwrap();

    println!("{:#?}", built_prompt);

    // Prints: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThis is a test prompt.<|eot_id|>

    // Alternatively, you can use the tokens for the prompt instead of the string.
    let _built_prompt_tokens = prompt.get_built_prompt_as_tokens().unwrap();
}
