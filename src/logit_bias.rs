use crate::tokenizer::LlmTokenizer;
use anyhow::{anyhow, Result};
use std::{collections::HashMap, sync::Arc};

/// Validates the logit bias token IDs by checking if each token ID can be converted to a single token.
///
/// # Arguments
///
/// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<(), anyhow::Error>` indicating success or an error if any of the token IDs are invalid.
pub fn validate_logit_bias_token_ids(
    tokenizer: &Arc<LlmTokenizer>,
    logit_bias: &HashMap<u32, f32>,
) -> Result<()> {
    for token_id in logit_bias.keys() {
        tokenizer.try_from_single_token_id(*token_id)?;
    }
    Ok(())
}

/// Converts logit biases from characters to a `HashMap` with token IDs as keys and bias values as values.
///
/// # Arguments
///
/// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with characters as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
pub fn logit_bias_from_chars(
    tokenizer: &Arc<LlmTokenizer>,
    logit_bias: &HashMap<char, f32>,
) -> Result<HashMap<u32, f32>> {
    let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
    for (char, bias) in logit_bias {
        let token_id = tokenizer.try_into_single_token(&char.to_string())?;
        token_logit_bias.insert(token_id, *bias);
    }
    Ok(token_logit_bias)
}

/// Converts logit biases from words to a `HashMap` with token IDs as keys and bias values as values.
/// If a word is longer than a single token, it will be split into multiple tokens.
/// Returns an error if the word is empty or contains whitespace.
///
/// # Arguments
///
/// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with words as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
pub fn logit_bias_from_words(
    tokenizer: &Arc<LlmTokenizer>,
    logit_bias: &HashMap<String, f32>,
) -> Result<HashMap<u32, f32>> {
    let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
    for (word_maybe, bias) in logit_bias {
        let mut words_maybe: Vec<String> = word_maybe
            .trim()
            .split_ascii_whitespace()
            .map(|s| s.trim().to_string())
            .collect();
        let word = if words_maybe.is_empty() {
            return Err(anyhow!(
                "logit_bias contains an empty word. Given word: {}",
                word_maybe
            ));
        } else if words_maybe.len() > 1 {
            return Err(anyhow!(
                "logit_bias contains a word seperated by whitespace. Given word: {}",
                word_maybe
            ));
        } else {
            words_maybe.remove(0)
        };
        let token_ids = tokenizer.tokenize(&word);
        for id in token_ids {
            if id == tokenizer.white_space_token_id {
                panic!(
                    "logit_bias contains a whitespace token. Given word: {}",
                    word
                )
            }
            token_logit_bias.insert(id, *bias);
        }
    }
    Ok(token_logit_bias)
}

/// Converts logit biases from texts to a `HashMap` with token IDs as keys and bias values as values.
/// This is for long texts that are tokenized into multiple tokens. It does not tokenize whitespace.
///
/// # Arguments
///
/// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with texts as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
pub fn logit_bias_from_texts(
    tokenizer: &Arc<LlmTokenizer>,
    logit_bias: &HashMap<String, f32>,
) -> Result<HashMap<u32, f32>> {
    let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
    for (text, bias) in logit_bias {
        let token_ids = tokenizer.tokenize(text);
        for id in token_ids {
            if id == tokenizer.white_space_token_id {
                continue;
            }
            token_logit_bias.insert(id, *bias);
        }
    }
    Ok(token_logit_bias)
}

/// Merges multiple logit biases into a single `HashMap` of token IDs and bias values.
///
/// # Arguments
///
/// * `logit_biases` - A vector of references to `HashMap`s containing logit biases with token IDs as keys and bias values as values.
///
/// # Returns
///
/// Returns a `HashMap<u32, f32>` containing the merged logit biases.
pub fn merge_logit_biases(logit_biases: Vec<&HashMap<u32, f32>>) -> HashMap<u32, f32> {
    let mut merged_logit_bias: HashMap<u32, f32> = HashMap::new();
    for logit_bias in logit_biases {
        for (token_id, bias) in logit_bias {
            merged_logit_bias.insert(*token_id, *bias);
        }
    }
    merged_logit_bias
}

/// Validates the logit bias values by checking if they are within the range of -100.0 to 100.0.
///
/// # Arguments
///
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<(), anyhow::Error>` indicating success or an error if any of the bias values are out of range.
pub fn validate_logit_bias_values(logit_bias: &HashMap<u32, f32>) -> Result<()> {
    for value in logit_bias.values() {
        if *value > 100.0 || *value < -100.0 {
            return Err(anyhow!(
                "logit_bias value must be between -100.0 and 100.0. Given value: {}",
                value
            ));
        }
    }
    Ok(())
}

/// Converts logit biases to the OpenAI format, where token IDs are represented as strings and bias values are rounded up to the nearest integer.
///
/// # Arguments
///
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
///
/// # Returns
///
/// Returns `Result<HashMap<String, serde_json::Value>, anyhow::Error>` containing the converted logit biases in the OpenAI format.
pub fn convert_logit_bias_to_openai_format(
    logit_bias: &HashMap<u32, f32>,
) -> HashMap<String, serde_json::Value> {
    let mut openai_logit_bias: HashMap<String, serde_json::Value> = HashMap::new();
    for (token_id, value) in logit_bias {
        openai_logit_bias.insert(
            token_id.to_string(),
            serde_json::Value::Number(serde_json::Number::from(value.ceil() as i32)),
        );
    }
    openai_logit_bias
}

/// Converts logit biases to the LLAMA format, where each logit bias is represented as a vector of two `serde_json::Value` elements: token ID and bias value.
///
/// # Arguments
///
/// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
///
/// # Returns
///
/// Returns a `Vec<Vec<serde_json::Value>>` containing the converted logit biases in the LLAMA format.
pub fn convert_logit_bias_to_llama_format(
    logit_bias: &HashMap<u32, f32>,
) -> Vec<Vec<serde_json::Value>> {
    let mut llama_logit_bias: Vec<Vec<serde_json::Value>> = Vec::new();
    for (token_id, value) in logit_bias {
        llama_logit_bias.push(vec![
            serde_json::Value::Number(serde_json::Number::from(*token_id)),
            serde_json::Value::Number(
                serde_json::Number::from_f64((*value).into()).expect("Invalid float value"),
            ),
        ]);
    }
    llama_logit_bias
}
