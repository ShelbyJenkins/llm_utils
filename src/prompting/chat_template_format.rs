use super::*;
use minijinja::{context, Environment, ErrorKind};

/// Applies a chat template to a message, given a message and a chat template.
///
/// # Arguments
///
/// * `message` - The message as a HashMap.
/// * `chat_template` - The chat template as a String.
///
/// # Returns
///
/// The formatted message as a String.
pub fn apply_chat_template(
    messages: &Vec<&HashMap<String, String>>,
    chat_template: &OsLlmChatTemplate,
) -> String {
    let mut env = Environment::new();
    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let template = &chat_template.chat_template;
    let bos_token = chat_template.bos_token.as_deref().unwrap_or("");
    let eos_token = chat_template.eos_token.as_deref().unwrap_or("");
    let unk_token = chat_template.unk_token.as_deref().unwrap_or("");

    // let template = chat_template.replace(".strip()", "|trim");
    env.add_template("chat_template", template)
        .expect("Failed to add template");
    env.add_function("raise_exception", raise_exception);
    let tmpl = env
        .get_template("chat_template")
        .expect("Failed to get template");
    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => bos_token,
        eos_token => eos_token,
        unk_token => unk_token,
    })
    .expect("Failed to render template")
}

/// This exists specifically for the minijinja template engine to raise an exception.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::open_source_model::preset::LlmPreset;

    const USER_PROMPT_1: &str = "tell me a joke";
    const ASSISTANT_PROMPT_1: &str = "the clouds";
    const USER_PROMPT_2: &str = "funny";
    const ASSISTANT_PROMPT_2: &str = "beepboop";
    const USER_PROMPT_3: &str = "robot?";

    #[test]
    fn test_chat_templates() -> Result<()> {
        let expected_outputs = [
                // meta-llama-3-8b-instruct
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\ntell me a joke<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nthe clouds<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nfunny<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nbeepboop<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nrobot?<|eot_id|>",
                // mistralai/Mistral-7B-Instruct-v0.3
                "<s>[INST] tell me a joke [/INST]the clouds</s>[INST] funny [/INST]beepboop</s>[INST] robot? [/INST]",
                // phi/Phi-3-mini-4k-instruct
                "<s><|user|>\ntell me a joke<|end|>\n<|assistant|>\nthe clouds<|end|>\n<|user|>\nfunny<|end|>\n<|assistant|>\nbeepboop<|end|>\n<|user|>\nrobot?<|end|>\n<|assistant|>\n",
        ];
        let messages: Vec<HashMap<String, String>> = vec![
            HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), USER_PROMPT_1.to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "assistant".to_string()),
                ("content".to_string(), ASSISTANT_PROMPT_1.to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), USER_PROMPT_2.to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "assistant".to_string()),
                ("content".to_string(), ASSISTANT_PROMPT_2.to_string()),
            ]),
            HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), USER_PROMPT_3.to_string()),
            ]),
        ];
        let templates = vec![
            LlmPreset::Llama3_8bInstruct.chat_template().unwrap(),
            LlmPreset::Mistral7bInstructV0_3.chat_template().unwrap(),
            LlmPreset::Phi3Mini4kInstruct.chat_template().unwrap(),
        ];

        for (i, chat_template) in templates.iter().enumerate() {
            let res = apply_chat_template(&messages.iter().collect::<Vec<_>>(), chat_template);

            assert_eq!(res, expected_outputs[i]);
        }
        Ok(())
    }
}
