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
    chat_template: &str,
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
fn apply_chat_template(message: HashMap<String, String>, chat_template: &str) -> Result<String> {
    let messages = vec![message];
    let mut env = Environment::new();
    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let template = chat_template.replace(".strip()", "|trim");
    env.add_template("chat_template", template.as_str())?;
    env.add_function("raise_exception", raise_exception);
    let tmpl = env.get_template("chat_template").unwrap();
    Ok(tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => "",
        eos_token => "",
        unk_token => "",
    })?)
}

/// Raises an exception with the specified message.
///
/// # Arguments
///
/// * `msg` - The exception message as a String.
///
/// # Returns
///
/// An error with the specified message.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYSTEM_MESSAGE: &str = "I'm listening1234";
    const USER_MESSAGE: &str = "Hello1234";

    #[test]
    fn test_chat_templates() -> Result<()> {
        let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
           "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            // mistralai/Mistral-7B-Instruct-v0.1
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            // meta-llama/Llama-2-13b-chat-hf
           "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            // google/gemma-7b-it
            "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
        ];
        let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>user\ninstructions: I'm listening1234\nuser input: Hello1234<|im_end|>\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "[INST] instructions: I'm listening1234\nuser input: Hello1234 [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "[INST] instructions: I'm listening1234\nuser input: Hello1234 [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "[INST] instructions: I'm listening1234\nuser input: Hello1234 [/INST]",
            // google/gemma-7b-it
            "<start_of_turn>user\ninstructions: I'm listening1234\nuser input: Hello1234<end_of_turn>\n",
        ];
        for (i, template) in templates.iter().enumerate() {
            let prompt = default_formatted_prompt(
                &Some(SYSTEM_MESSAGE.to_string()),
                &None,
                &Some(USER_MESSAGE.to_string()),
            )?;
            let res = convert_default_prompt_to_model_format(&prompt, template)?;
            assert_eq!(res, expected_outputs[i]);
        }
        Ok(())
    }
}
