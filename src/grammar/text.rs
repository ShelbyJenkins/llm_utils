use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone)]
pub struct TextGrammar {
    pub item_token_length: u32,
    pub stop_word_done: Option<String>,
    pub stop_word_null_result: Option<String>,
    grammar_string: RefCell<Option<String>>,
}

impl Default for TextGrammar {
    fn default() -> Self {
        Self {
            item_token_length: 200,
            stop_word_done: None,
            stop_word_null_result: None,
            grammar_string: RefCell::new(None),
        }
    }
}

impl TextGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Text(self)
    }

    pub fn item_token_length(mut self, item_token_length: u32) -> Self {
        self.item_token_length = item_token_length;
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(text_grammar(
                self.item_token_length,
                &self.stop_word_done,
                &self.stop_word_null_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        text_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        text_parse(content)
    }
}

impl GrammarSetterTrait for TextGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_null_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_null_result
    }
}

const CHAR_NO_NEWLINE: &str = r"[^\r\x0b\x0c\x85\u2028\u2029]";
// const CHAR_NO_NEWLINE: &str = r"[^\r\n\x0b\x0c\x85\u2028\u2029]";
pub fn text_grammar(
    item_token_length: u32,
    stop_word_done: &Option<String>,
    stop_word_null_result: &Option<String>,
) -> String {
    match (stop_word_done, stop_word_null_result) {
        (Some(stop_word_done), Some(stop_word_null_result)) => {
            format!(
                "root ::= \" \" ( item{{1,{}}} | \"{stop_word_null_result}\" ) \" {stop_word_done}\"\nitem ::= {CHAR_NO_NEWLINE}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" item{{1,{}}} \" {stop_word_done}\"\nitem ::= {CHAR_NO_NEWLINE}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
        (None, Some(stop_word_null_result)) => {
            format!(
                "root ::= \" \" ( item{{1,{}}} | \"{stop_word_null_result}\" )\nitem ::= {CHAR_NO_NEWLINE}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
        (None, None) => {
            format!(
                "root ::= \" \" item{{0,{}}}\n\nitem ::= {CHAR_NO_NEWLINE}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
    }
}

pub fn text_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches(|c: char| !(c.is_alphanumeric() || c.is_ascii_punctuation()));

    if text_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        })
    }
}

pub fn text_parse(content: &str) -> Result<String, GrammarError> {
    if content.is_empty() {
        return Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        });
    }
    Ok(content.to_string())
}