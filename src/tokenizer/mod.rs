use crate::hf_loader::HuggingFaceLoader;
use anyhow::{anyhow, Result};
use std::{fmt, path::PathBuf};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tokenizers::Tokenizer as HFTokenizer;

pub enum TokenizerBackend {
    HuggingFacesTokenizer(HFTokenizer),
    Tiktoken(CoreBPE),
}

impl fmt::Debug for TokenizerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerBackend::HuggingFacesTokenizer(_) => {
                write!(f, "TokenizerBackend::HuggingFacesTokenizer")
            }
            TokenizerBackend::Tiktoken(_) => {
                write!(f, "TokenizerBackend::Tiktoken")
            }
        }
    }
}

#[derive(Debug)]
pub struct LlmTokenizer {
    pub tokenizer: TokenizerBackend,
    pub tokenizer_path: Option<PathBuf>,
    pub with_special_tokens: bool,
    pub white_space_token_id: u32,
}

impl LlmTokenizer {
    pub fn new_tiktoken(model_id: &str) -> Self {
        let tokenizer = get_bpe_from_model(model_id).unwrap();
        let white_space_token_id = tokenizer.encode_ordinary(" ").remove(0) as u32;
        Self {
            tokenizer: TokenizerBackend::Tiktoken(tokenizer),
            tokenizer_path: None,
            with_special_tokens: false,
            white_space_token_id,
        }
    }

    pub fn new_from_tokenizer_json(tokenizer_json_path: &PathBuf) -> Self {
        let tokenizer = HFTokenizer::from_file(tokenizer_json_path).unwrap();
        let white_space_token_id = tokenizer.encode(" ", false).unwrap().get_ids()[0];
        Self {
            tokenizer: TokenizerBackend::HuggingFacesTokenizer(tokenizer),
            tokenizer_path: Some(tokenizer_json_path.clone()),
            with_special_tokens: false,
            white_space_token_id,
        }
    }

    pub fn new_from_hf_repo(hf_token: &Option<String>, repo_id: &str) -> Result<Self> {
        let hf_loader = HuggingFaceLoader::new(hf_token.clone()).model_from_repo_id(repo_id);
        let tokenizer_json_path: PathBuf = hf_loader.load_file("tokenizer.json")?;
        Ok(LlmTokenizer::new_from_tokenizer_json(&tokenizer_json_path))
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

    /// Creates a window of text normalized to the specified token size in the center of the text.
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

    /// Creates a range of text from the specified start and end token indices.
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
    pub fn create_text_range(
        &self,
        text: &str,
        start_token_index: u32,
        end_token_index: u32,
    ) -> String {
        let tokens = self.tokenize(text);
        let end_token_index = if tokens.len() <= end_token_index as usize {
            tokens.len()
        } else {
            end_token_index as usize
        };

        let preserved_tokens = &tokens[start_token_index as usize..end_token_index];
        self.detokenize_many(preserved_tokens).unwrap()
    }

    pub fn chunk_text(
        &self,
        text: &str,
        max_length: usize,
        overlap_percent: Option<usize>,
    ) -> Vec<String> {
        let mut splitter =
            crate::text_utils::chunk::DFSTextSplitter::new(max_length, overlap_percent, self);
        splitter.run(text).unwrap()
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
