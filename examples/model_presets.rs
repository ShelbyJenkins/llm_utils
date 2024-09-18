use gguf::GgufLoader;
use llm_utils::{models::local_model::*, prompting::LlmPrompt};

fn main() {
    // Using
    let model: LocalLlmModel = GgufLoader::default()
        .llama3_1_8b_instruct()
        .available_vram(48)
        .use_ctx_size(9001)
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
