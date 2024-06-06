use crate::models::open_source::OsLlmChatTemplate;
use anyhow::{anyhow, Result};
use minijinja::{context, Environment, ErrorKind};
use std::{collections::HashMap, fs::File, io::Read, path::Path};

/// Returns a formatted prompt as a HashMap, given optional system content, system content path, and user content. This is the Openai spec.
///
/// # Arguments
///
/// * `system_content` - Optional system content as a String.
/// * `system_content_path` - Optional system content path as a String.
/// * `user_content` - Optional user content as a String.
///
/// # Returns
///
/// A HashMap representing the formatted prompt, with "system" and "user" prompts stored as key-value pairs.
///
/// # Errors
///
/// Returns an error if none of the input parameters are set.
pub fn default_formatted_prompt(
    system_content: &Option<String>,
    system_content_path: &Option<String>,
    user_content: &Option<String>,
) -> Result<HashMap<String, HashMap<String, String>>, anyhow::Error> {
    if system_content.is_none() && system_content_path.is_none() && user_content.is_none() {
        return Err(anyhow!(
            "system_content, system_content_path, or user_content must be set to build a prompt."
        ));
    }
    let system_prompt = create_system_prompt(system_content, system_content_path);
    let user_prompt = create_user_prompt(user_content);
    Ok(HashMap::from([
        ("system".to_string(), system_prompt),
        ("user".to_string(), user_prompt),
    ]))
}

/// Returns a HashMap representing the system prompt, given optional system content and system content path.
///
/// # Arguments
///
/// * `system_content` - Optional system content as a String.
/// * `system_content_path` - Optional system content path as a String.
///
/// # Returns
///
/// A HashMap representing the system prompt, with "content" and "name" fields.
fn create_system_prompt(
    system_content: &Option<String>,
    system_content_path: &Option<String>,
) -> HashMap<String, String> {
    let mut system_content_str: String = String::new();
    if let Some(system_content) = system_content {
        system_content_str.push_str(&system_content.to_string());
    }
    if let Some(system_content_path) = system_content_path {
        if !system_content_str.is_empty() {
            system_content_str.push('\n');
        };
        system_content_str.push_str(&load_system_content_path(system_content_path).to_string());
    }
    HashMap::from([
        ("content".to_string(), system_content_str),
        ("name".to_string(), "".to_string()),
    ])
}

/// Returns the content loaded from the file at the specified system content path. Uses a YAML file.
///
/// # Arguments
///
/// * `system_content_path` - The system content path as a String.
///
/// # Returns
///
/// The content loaded from the file as a String.
///
/// # Panics
///
/// Panics if the file is empty.
fn load_system_content_path(system_content_path: &str) -> String {
    let path = Path::new(&system_content_path);
    match File::open(path) {
        Ok(mut file) => {
            let mut custom_prompt = String::new();
            match file.read_to_string(&mut custom_prompt) {
                Ok(_) => {
                    if custom_prompt.trim().is_empty() {
                        panic!("system_content_path '{}' is empty.", path.display())
                    }
                    custom_prompt
                }
                Err(e) => panic!("Failed to read file: {}", e),
            }
        }
        Err(e) => panic!("Failed to open file: {}", e),
    }
}

/// Returns a HashMap representing the user prompt, given optional user content.
///
/// # Arguments
///
/// * `user_content` - Optional user content as a String.
///
/// # Returns
///
/// A HashMap representing the user prompt, with "content" and "name" fields.
fn create_user_prompt(user_content: &Option<String>) -> HashMap<String, String> {
    let mut user_content_str: String = String::new();
    if let Some(user_content) = user_content {
        user_content_str.push_str(&user_content.to_string());
    }
    HashMap::from([
        ("content".to_string(), user_content_str),
        ("name".to_string(), "".to_string()),
    ])
}

/// Converts the default prompt to a String. Used when an LLMBackend applies the chat template.
///
/// # Arguments
///
/// * `default_prompt` - The default prompt as a HashMap.
///
/// # Returns
///
/// The prompt converted to a String.
pub fn convert_default_prompt_to_string(
    default_prompt: &HashMap<String, HashMap<String, String>>,
) -> String {
    let mut content_str = String::new();
    if let Some(system_content) = default_prompt
        .get("system")
        .and_then(|system| system.get("content"))
    {
        if !system_content.is_empty() {
            content_str.push_str(&format!("instructions: {}\n", system_content));
        }
    }
    if let Some(user_content) = default_prompt
        .get("user")
        .and_then(|user| user.get("content"))
    {
        if !user_content.is_empty() {
            content_str.push_str(&format!("user input: {}", user_content));
        }
    };

    content_str
}

/// Converts the default prompt to a model-specific format, given a default prompt and a chat template.
///
/// # Arguments
///
/// * `default_prompt` - The default prompt as a HashMap.
/// * `chat_template` - The chat template as a String.
///
/// # Returns
///
/// The prompt converted to a model-specific format as a String.
pub fn convert_default_prompt_to_model_format(
    default_prompt: &HashMap<String, HashMap<String, String>>,
    chat_template: &OsLlmChatTemplate,
) -> Result<String> {
    let mut content_str = String::new();
    if let Some(system_content) = default_prompt
        .get("system")
        .and_then(|system| system.get("content"))
    {
        if !system_content.is_empty() {
            content_str.push_str(&format!("instructions: {}\n", system_content));
        }
    }
    if let Some(user_content) = default_prompt
        .get("user")
        .and_then(|user| user.get("content"))
    {
        if !user_content.is_empty() {
            content_str.push_str(&format!("user input: {}", user_content));
        }
    };
    let preformated_prompt = HashMap::from([
        ("role".to_string(), "user".to_string()),
        ("content".to_string(), content_str.to_string()),
    ]);
    apply_chat_template(preformated_prompt, chat_template)
}

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
fn apply_chat_template(
    message: HashMap<String, String>,
    chat_template: &OsLlmChatTemplate,
) -> Result<String> {
    let messages = vec![message];
    let mut env = Environment::new();
    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let template = &chat_template.chat_template;
    let bos_token = chat_template.bos_token.as_deref().unwrap_or("");
    let eos_token = chat_template.eos_token.as_deref().unwrap_or("");
    let unk_token = chat_template.unk_token.as_deref().unwrap_or("");

    // let template = chat_template.replace(".strip()", "|trim");
    env.add_template("chat_template", template)?;
    env.add_function("raise_exception", raise_exception);
    let tmpl = env.get_template("chat_template").unwrap();
    Ok(tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => bos_token,
        eos_token => eos_token,
        unk_token => unk_token,
    })?)
}

/// This exists specifically for the minijinja template engine to raise an exception.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

/// Sets and validates the max_tokens/n_predict parameter for a request. First, calculates the available tokens for a model after accounting for the prompt length.  
/// Then it use either the model_token_utilization as a percentage of available tokens, the directly specified requested_response_tokens, or if neither are set, it defaults to max_tokens_output_for_model.
/// Attempts to reduce the requested_response_tokens if it exceeds the available tokens.
/// Validates that the requested_response_tokens + total_prompt_tokens is less than context_length_for_model.
/// Finally, validates that the requested_response_tokens is less than max_tokens_output_for_model.
///
/// # Arguments
///
/// * `context_length_for_model` - The context length for the model as an unsigned 32-bit integer.
/// * `max_tokens_output_for_model` - The maximum tokens output for the model as an unsigned 32-bit integer.
/// * `total_prompt_tokens` - The total prompt tokens as an unsigned 32-bit integer.
/// * `safety_tokens` - The safety tokens as an unsigned 32-bit integer.
/// * `model_token_utilization` - The model token utilization as an optional floating-point number.
/// * `requested_response_tokens` - The requested response tokens as an optional unsigned 32-bit integer.
///
/// # Returns
///
/// The maximum number of tokens allowed for the response as an unsigned 32-bit integer.
///
/// # Errors
///
/// Returns an error if any of the validation checks fail.
pub fn get_and_check_max_tokens_for_response(
    context_length_for_model: u32,
    max_tokens_output_for_model: u32,
    total_prompt_tokens: u32,
    safety_tokens: u32,
    model_token_utilization: Option<f32>,
    requested_response_tokens: Option<u32>,
) -> Result<u32> {
    if total_prompt_tokens >= context_length_for_model {
        return Err(anyhow!(
            "total_prompt_tokens is greater than context_length_for_model. total_prompt_tokens: {}, context_length_for_model: {}",
            total_prompt_tokens,
            context_length_for_model
        ));
    }

    let available_tokens = std::cmp::min(
        context_length_for_model - total_prompt_tokens,
        max_tokens_output_for_model,
    );

    let mut requested_response_tokens =
        if let Some(model_token_utilization) = model_token_utilization {
            (available_tokens as f32 * (model_token_utilization)).ceil() as u32
        } else if let Some(requested_response_tokens) = requested_response_tokens {
            requested_response_tokens
        } else {
            max_tokens_output_for_model
        };
    while requested_response_tokens > (available_tokens - safety_tokens) {
        requested_response_tokens -= 1
    }
    if requested_response_tokens == 0 {
        return Err(anyhow!("after validation actual_response_tokens is 0"));
    }
    if requested_response_tokens > max_tokens_output_for_model {
        return Err(anyhow!(
            "requested_response_tokens is greater than max_tokens_output_for_model. requested_response_tokens: {}, than max_tokens_output_for_model: {}",
            requested_response_tokens,
            max_tokens_output_for_model
        ));
    }
    if requested_response_tokens + total_prompt_tokens >= context_length_for_model {
        return Err(anyhow!(
            "requested_response_tokens + total_prompt_tokens is greater than context_length_for_model. requested_response_tokens: {}, total_prompt_tokens: {}, context_length_for_model: {}",
            requested_response_tokens,
            total_prompt_tokens,
            context_length_for_model
        ));
    }
    Ok(requested_response_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYSTEM_MESSAGE: &str = "I'm listening";
    const USER_MESSAGE: &str = "Hello!";

    #[test]
    fn test_chat_templates() -> Result<()> {
        let templates = [
            // mistralai/Mistral-7B-Instruct-v0.3
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            // meta-llama-3-8b-instruct
            "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            // google/gemma-7b-it
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
           "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            ];
        let expected_outputs = [
                // mistralai/Mistral-7B-Instruct-v0.1
                "<s>[INST] instructions: I'm listening\nuser input: Hello! [/INST]",
                // meta-llama-3-8b-instruct
                "<s><|start_header_id|>user<|end_header_id|>\n\ninstructions: I'm listening\nuser input: Hello!<|eot_id|>",
                // mistralai/Mixtral-8x7B-Instruct-v0.1
                "<s>[INST] instructions: I'm listening\nuser input: Hello! [/INST]",
                // google/gemma-7b-it
                "<s><start_of_turn>user\ninstructions: I'm listening\nuser input: Hello!<end_of_turn>\n",
                // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
                "<|im_start|>user\ninstructions: I'm listening\nuser input: Hello!<|im_end|>\n",
        ];
        for (i, template) in templates.iter().enumerate() {
            let prompt = default_formatted_prompt(
                &Some(SYSTEM_MESSAGE.to_string()),
                &None,
                &Some(USER_MESSAGE.to_string()),
            )?;
            let chat_template = OsLlmChatTemplate {
                chat_template: template.to_string(),
                chat_template_path: None,
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                unk_token: Some("<unk>".to_string()),
            };

            let res = convert_default_prompt_to_model_format(&prompt, &chat_template)?;
            assert_eq!(res, expected_outputs[i]);
        }
        Ok(())
    }
}
