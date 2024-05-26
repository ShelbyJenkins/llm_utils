use anyhow::{anyhow, Result};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tokenizers::Tokenizer as HFTokenizer;

pub enum TokenizerBackend {
    HuggingFacesTokenizer(HFTokenizer),
    Tiktoken(CoreBPE),
}

pub struct LlmUtilsTokenizer {
    pub tokenizer: TokenizerBackend,
    pub with_special_tokens: bool,
    pub white_space_token_id: u32,
}

impl LlmUtilsTokenizer {
    pub fn new_tiktoken(model_id: &str) -> Self {
        let tokenizer = get_bpe_from_model(model_id).unwrap();
        let white_space_token_id = tokenizer.encode_ordinary(" ").remove(0) as u32;
        Self {
            tokenizer: TokenizerBackend::Tiktoken(tokenizer),
            with_special_tokens: false,
            white_space_token_id,
        }
    }

    pub fn new_from_model(tokenizer: HFTokenizer) -> Self {
        let white_space_token_id = tokenizer.encode(" ", false).unwrap().get_ids()[0];
        Self {
            tokenizer: TokenizerBackend::HuggingFacesTokenizer(tokenizer),
            with_special_tokens: false,
            white_space_token_id,
        }
    }

    pub fn tokenize(&self, str: &str) -> Vec<u32> {
        self.encode(str)
    }

    pub fn detokenize_one(&self, token: u32) -> Result<String> {
        self.decode(&[token])
    }

    pub fn detokenize_many(&self, tokens: &[u32]) -> Result<String> {
        self.decode(tokens)
    }

    pub fn count_tokens(&self, str: &str) -> u32 {
        let tokens = self.tokenize(str);
        tokens.len() as u32
    }

    pub fn try_from_single_token_id(&self, try_from_single_token_id: u32) -> Result<String> {
        let detokenize_response = self.detokenize_one(try_from_single_token_id)?;
        println!("detokenize_response: {}", detokenize_response);
        let mut strings_maybe: Vec<String> = detokenize_response
            .split_ascii_whitespace()
            .map(|s| s.to_string())
            .collect();
        match strings_maybe.len() {
            0 => Err(anyhow!(
                "token_id is empty for try_from_single_token_id: {}",
                try_from_single_token_id
            )),
            1 => Ok(strings_maybe.remove(0)),
            n => Err(anyhow!(
                "Found more than one token ({n} total) in try_from_single_token_id: {}",
                try_from_single_token_id
            )),
        }
    }

    pub fn try_into_single_token(&self, try_into_single_token: &str) -> Result<u32> {
        let mut tokens = self.tokenize(try_into_single_token);
        match tokens.len() {
            0 => Err(anyhow!("No token found in text: {}", try_into_single_token)),
            1 => Ok(tokens.remove(0)),
            n => Err(anyhow!(
                "Found more than one token ({n} total) in text: {}",
                try_into_single_token
            )),
        }
    }

    /// Creates a window of text normalized to the specified token size.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to create a window from.
    /// * `target_token_size` - The desired number of tokens in the window.
    ///
    /// # Returns
    ///
    /// A new string that represents the normalized window of text, or the original
    /// text if its token count is less than or equal to `target_token_size`.
    pub fn create_text_window(&self, text: &str, target_token_size: u32) -> String {
        let tokens = self.tokenize(text);
        if tokens.len() <= target_token_size as usize {
            return text.to_string();
        }

        let start_token_index = (tokens.len() - target_token_size as usize) / 2;
        let end_token_index = start_token_index + target_token_size as usize;

        let preserved_tokens = &tokens[start_token_index..end_token_index];
        self.detokenize_many(preserved_tokens).unwrap()
    }

    fn encode_tiktoken(&self, tokenizer: &CoreBPE, str: &str) -> Vec<u32> {
        let tokens = if self.with_special_tokens {
            tokenizer
                .encode_with_special_tokens(str)
                .iter()
                .map(|&x| x as u32)
                .collect()
        } else {
            tokenizer
                .encode_ordinary(str)
                .iter()
                .map(|&x| x as u32)
                .collect()
        };
        tokens
    }

    fn encode_hf(&self, tokenizer: &HFTokenizer, str: &str) -> Vec<u32> {
        let tokens = if self.with_special_tokens {
            tokenizer.encode(str, true)
        } else {
            tokenizer.encode(str, false)
        };
        tokens.unwrap().get_ids().to_vec()
    }

    fn encode(&self, str: &str) -> Vec<u32> {
        match &self.tokenizer {
            TokenizerBackend::HuggingFacesTokenizer(tokenizer) => self.encode_hf(tokenizer, str),
            TokenizerBackend::Tiktoken(tokenizer) => self.encode_tiktoken(tokenizer, str),
        }
    }

    fn decode_tiktoken(&self, tokenizer: &CoreBPE, tokens: &[u32]) -> Result<String> {
        tokenizer
            .decode(tokens.iter().map(|&x| x as usize).collect())
            .map_err(|e| anyhow!(e))
    }

    fn decode_hf(&self, tokenizer: &HFTokenizer, tokens: &[u32]) -> Result<String> {
        tokenizer.decode(tokens, true).map_err(|e| anyhow!(e))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match &self.tokenizer {
            TokenizerBackend::HuggingFacesTokenizer(tokenizer) => self.decode_hf(tokenizer, tokens),
            TokenizerBackend::Tiktoken(tokenizer) => self.decode_tiktoken(tokenizer, tokens),
        }
    }
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
