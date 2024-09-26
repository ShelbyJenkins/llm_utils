use super::{PromptMessage, TextConcatenator};
use crate::models::api_model::ApiLlmModel;
use std::collections::HashMap;

#[derive(Clone)]
pub struct OpenAiPrompt {
    pub built_prompt_hashmap: std::cell::RefCell<Option<Vec<HashMap<String, String>>>>,
    pub total_prompt_tokens: std::cell::RefCell<Option<u64>>,
    pub concatenator: TextConcatenator,
    pub messages: std::cell::RefCell<Vec<PromptMessage>>,
    pub api_llm: ApiLlmModel,
}

impl OpenAiPrompt {
    pub fn new(api_llm: &ApiLlmModel) -> OpenAiPrompt {
        OpenAiPrompt {
            built_prompt_hashmap: std::cell::RefCell::new(None),
            total_prompt_tokens: std::cell::RefCell::new(None),
            concatenator: TextConcatenator::default(),
            messages: std::cell::RefCell::new(Vec::new()),
            api_llm: api_llm.clone(),
        }
    }

    pub fn build_prompt(&self) -> Vec<HashMap<String, String>> {
        self.clear_built_prompt();
        let built_prompt_hashmap =
            super::prompt_message::build_messages(&mut self.messages.borrow_mut());
        *self.total_prompt_tokens.borrow_mut() =
            Some(super::token_count::total_prompt_tokens_openai_format(
                &built_prompt_hashmap,
                self.api_llm.tokens_per_message,
                self.api_llm.tokens_per_name,
                &self.api_llm.model_base.tokenizer,
            ));
        *self.built_prompt_hashmap.borrow_mut() = Some(built_prompt_hashmap.clone());
        built_prompt_hashmap
    }

    pub fn clear_built_prompt(&self) {
        *self.built_prompt_hashmap.borrow_mut() = None;
        *self.total_prompt_tokens.borrow_mut() = None;
    }
}

impl std::fmt::Display for OpenAiPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "OpenAiPrompt")?;
        for message in self.messages.borrow().iter() {
            writeln!(f, "{}", message)?;
        }

        match *self.built_prompt_hashmap.borrow() {
            Some(ref prompt) => {
                writeln!(f, "built_prompt_hashmap:\n{:?}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "built_prompt_hashmap: None")?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompting::LlmPrompt;

    #[test]
    fn test_openai() {
        let model = ApiLlmModel::gpt_3_5_turbo();
        let prompt = LlmPrompt::new_openai_prompt(&model);

        prompt
            .add_system_message()
            .unwrap()
            .set_content("test system content");
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

        let test_openai = prompt.get_built_prompt_hashmap().unwrap();
        println!("{prompt}");
        let result_openai = vec![
            HashMap::from([
                ("content".to_string(), "test system content".to_string()),
                ("role".to_string(), "system".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test user content 1".to_string()),
                ("role".to_string(), "user".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test assistant content".to_string()),
                ("role".to_string(), "assistant".to_string()),
            ]),
            HashMap::from([
                ("content".to_string(), "test user content 2".to_string()),
                ("role".to_string(), "user".to_string()),
            ]),
        ];
        assert_eq!(test_openai, result_openai);
        // model
        // .openai_token_count_of_prompt(&test_openai)?,
        // let token_count = prompt.get_total_prompt_tokens().unwrap();
        // assert_eq!(31, token_count);
    }
}
