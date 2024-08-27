use llm_utils::{models::open_source_model::*, prompting::LlmPrompt};

fn main() {
    // Using
    let model: OsLlm = OsLlmLoader::new()
        .llama3_8b_instruct()
        .available_vram(48)
        .use_ctx_size(9001)
        .load()
        .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.

    let mut prompt: LlmPrompt = LlmPrompt::new_from_os_llm(&model);

    prompt
        .add_user_message()
        .set_content("This is a test prompt.");

    prompt.build().unwrap();

    // To feed into the inference engine
    let built_prompt = prompt.built_chat_template_prompt.clone().unwrap();

    println!("{:#?}", built_prompt);

    // Prints: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThis is a test prompt.<|eot_id|>

    // Alternatively, you can use the tokens for the prompt instead of the string.
    let _built_prompt_tokens = prompt.built_prompt_as_tokens.clone().unwrap();
}
