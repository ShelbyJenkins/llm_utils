use super::{PromptMessage, TextConcatenator};
use crate::{
    models::local_model::{LlmChatTemplate, LocalLlmModel},
    tokenizer::LlmTokenizer,
};
use minijinja::{context, Environment, ErrorKind};
use std::collections::HashMap;

#[derive(Clone)]
pub struct ChatTemplatePrompt {
    pub built_prompt_string: std::cell::RefCell<Option<String>>,
    pub built_prompt_as_tokens: std::cell::RefCell<Option<Vec<u32>>>,
    pub total_prompt_tokens: std::cell::RefCell<Option<u32>>,
    pub concatenator: TextConcatenator,
    pub chat_template: LlmChatTemplate,
    pub messages: std::cell::RefCell<Vec<PromptMessage>>,
    pub generation_prefix: std::cell::RefCell<Option<String>>,
    pub tokenizer: std::sync::Arc<LlmTokenizer>,
}

impl ChatTemplatePrompt {
    pub fn new(local_llm: &LocalLlmModel) -> ChatTemplatePrompt {
        ChatTemplatePrompt {
            built_prompt_string: std::cell::RefCell::new(None),
            built_prompt_as_tokens: std::cell::RefCell::new(None),
            total_prompt_tokens: std::cell::RefCell::new(None),
            concatenator: TextConcatenator::default(),
            chat_template: local_llm.chat_template.clone(),
            messages: std::cell::RefCell::new(Vec::new()),
            generation_prefix: std::cell::RefCell::new(None),
            tokenizer: local_llm.model_base.tokenizer.clone(),
        }
    }

    // Setter functions
    pub fn set_generation_prefix<T: AsRef<str>>(&self, generation_prefix: T) {
        if self.generation_prefix.borrow().is_none()
            || self.generation_prefix.borrow().as_deref() != Some(generation_prefix.as_ref())
        {
            self.clear_built_prompt();
            *self.generation_prefix.borrow_mut() = Some(generation_prefix.as_ref().to_string());
        }
    }

    pub fn clear_generation_prefix(&self) {
        self.clear_built_prompt();
        *self.generation_prefix.borrow_mut() = None;
    }

    pub fn build_prompt(&self) -> String {
        self.clear_built_prompt();
        let prompt_messages =
            super::prompt_message::build_messages(&mut self.messages.borrow_mut());

        let mut built_prompt_string = apply_chat_template(
            &prompt_messages,
            &self.chat_template.chat_template,
            &self.chat_template.bos_token,
            &self.chat_template.eos_token,
            self.chat_template.unk_token.as_deref(),
        );

        if let Some(ref generation_prefix) = *self.generation_prefix.borrow() {
            if let Some(base_generation_prefix) = &self.chat_template.base_generation_prefix {
                built_prompt_string.push_str(base_generation_prefix);
            }
            built_prompt_string.push_str(generation_prefix);
        }

        let built_prompt_as_tokens = self.tokenizer.tokenize(&built_prompt_string);
        *self.total_prompt_tokens.borrow_mut() = Some(built_prompt_as_tokens.len() as u32);
        *self.built_prompt_as_tokens.borrow_mut() = Some(built_prompt_as_tokens);
        *self.built_prompt_string.borrow_mut() = Some(built_prompt_string.clone());
        built_prompt_string
    }

    pub fn clear_built_prompt(&self) {
        *self.built_prompt_string.borrow_mut() = None;
        *self.built_prompt_as_tokens.borrow_mut() = None;
        *self.total_prompt_tokens.borrow_mut() = None;
    }
}

impl std::fmt::Display for ChatTemplatePrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ChatTemplatePrompt")?;
        for message in self.messages.borrow().iter() {
            writeln!(f, "{}", message)?;
        }

        match *self.built_prompt_string.borrow() {
            Some(ref prompt) => {
                writeln!(f, "built_prompt_string:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "built_prompt_string: None")?,
        };

        match *self.total_prompt_tokens.borrow() {
            Some(ref prompt) => {
                writeln!(f, "total_prompt_tokens:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "total_prompt_tokens: None")?,
        };

        Ok(())
    }
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
pub fn apply_chat_template(
    messages: &Vec<HashMap<String, String>>,
    chat_template: &str,
    bos_token: &str,
    eos_token: &str,
    unk_token: Option<&str>,
) -> String {
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_template("chat_template", chat_template)
        .expect("Failed to add template");
    env.add_function("raise_exception", raise_exception);

    let tmpl = env
        .get_template("chat_template")
        .expect("Failed to get template");

    let unk_token = unk_token.unwrap_or("");

    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => bos_token,
        eos_token => eos_token,
        unk_token => unk_token,
    })
    .expect("Failed to render template without system prompt")
}

/// This exists specifically for the minijinja template engine to raise an exception.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[cfg(test)]
mod tests {

    use crate::{
        models::local_model::{preset::LlmPreset, LocalLlmModel},
        prompting::{chat_template_prompt::apply_chat_template, LlmPrompt},
    };
    use std::collections::HashMap;

    #[test]
    fn test_chat() {
        let model = LocalLlmModel::default();
        let prompt = LlmPrompt::new_chat_template_prompt(&model);

        prompt
            .add_user_message()
            .unwrap()
            .set_content("test user content 1");
        prompt
            .add_assistant_message()
            .unwrap()
            .set_content("test assistant content");
        prompt
            .add_user_message()
            .unwrap()
            .set_content("test user content 2");

        let test_chat = prompt.get_built_prompt_string().unwrap();
        println!("{prompt}",);
        assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|>"
        );
        let token_count = prompt.get_total_prompt_tokens().unwrap();
        let prompt_as_tokens = prompt.get_built_prompt_as_tokens().unwrap();
        assert_eq!(54, token_count);
        assert_eq!(token_count, prompt_as_tokens.len() as u32);

        prompt.set_generation_prefix("Generating 12345:");
        let test_chat = prompt.get_built_prompt_string().unwrap();
        println!("{prompt}");
        assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerating 12345:"
        );
        let token_count = prompt.get_total_prompt_tokens().unwrap();
        let prompt_as_tokens = prompt.get_built_prompt_as_tokens().unwrap();
        assert_eq!(63, token_count);
        assert_eq!(token_count, prompt_as_tokens.len() as u32);
    }

    const USER_PROMPT_1: &str = "tell me a joke";
    const ASSISTANT_PROMPT_1: &str = "the clouds";
    const USER_PROMPT_2: &str = "funny";
    const ASSISTANT_PROMPT_2: &str = "beepboop";
    const USER_PROMPT_3: &str = "robot?";

    #[test]
    fn test_chat_templates() {
        let expected_outputs = [
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
            LlmPreset::Mistral7bInstructV0_3
                .load()
                .unwrap()
                .chat_template,
            LlmPreset::Phi3Mini4kInstruct.load().unwrap().chat_template,
        ];

        for (i, chat_template) in templates.iter().enumerate() {
            let res = apply_chat_template(
                &messages,
                &chat_template.chat_template,
                &chat_template.bos_token,
                &chat_template.eos_token,
                chat_template.unk_token.as_deref(),
            );

            assert_eq!(res, expected_outputs[i]);
        }
    }
}
