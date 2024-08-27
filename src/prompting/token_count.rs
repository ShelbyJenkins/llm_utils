use super::*;
use thiserror::Error;

pub const DEFAULT_SAFETY_TOKENS: u32 = 10;

/// Sets and validates the 'max_tokens' or 'n_ctx' or 'n_predict' parameter for a request.
/// First, it checks that the total_prompt_tokens is less than the ctx_size - safety_tokens.
/// Then returns 'available_tokens' as the lower of either:
/// ctx_size - total_prompt_tokens - safety_tokens or if it's provided, ctx_output_size.
/// If 'requested_tokens' is provided, 'requested_tokens' is returned if less than 'available_tokens'.
/// If 'requested_tokens' is 'None' or 'requested_tokens' is greater than 'available_tokens',
/// 'available_tokens' is returned.
///
/// # Arguments
///
/// * `ctx_size` - The total context length for the for the model or system.
/// * `ctx_output_size` - Optional output size for models with output generation limits. Defaults to None.
/// * `total_prompt_tokens` - The total prompt tokens as an unsigned 32-bit integer.
/// * `safety_tokens` - Optional padding. Defaults to 10.
/// * `requested_tokens` - Optional 'max_tokens' for the response. Defaults to 'available_tokens'.
///
/// # Returns
///
/// A u32 to be used for the 'max_tokens' or 'n_ctx' parameter for inference requests.
///
/// # Errors
///
/// Returns an error if any of the validation checks fail.
pub fn check_and_get_max_tokens(
    ctx_size: u32,
    ctx_output_size: Option<u32>,
    total_prompt_tokens: u32,
    safety_tokens: Option<u32>,
    requested_tokens: Option<u32>,
) -> Result<u32, RequestTokenLimitError> {
    let available_tokens = available_tokens(
        ctx_size,
        ctx_output_size,
        total_prompt_tokens,
        safety_tokens,
    )?;
    let requested_tokens = if let Some(requested_tokens) = requested_tokens {
        if requested_tokens > available_tokens {
            eprintln!(
                "requested_tokens ({requested_tokens}) is greater than available_tokens ({}). Using available_tokens for request.", available_tokens
            );
            available_tokens
        } else {
            requested_tokens
        }
    } else {
        available_tokens
    };

    if total_prompt_tokens + requested_tokens >= ctx_size {
        panic!(
            "total_prompt_tokens ({total_prompt_tokens}) + requested_tokens ({requested_tokens}) >= ctx_size ({ctx_size}). This should never happen.",
        );
    }
    Ok(requested_tokens)
}

pub fn available_tokens(
    ctx_size: u32,
    ctx_output_size: Option<u32>,
    total_prompt_tokens: u32,
    safety_tokens: Option<u32>,
) -> Result<u32, RequestTokenLimitError> {
    let safety_tokens = safety_tokens.unwrap_or(DEFAULT_SAFETY_TOKENS);

    if total_prompt_tokens >= ctx_size - safety_tokens {
        return Err(RequestTokenLimitError::PromptTokensExceeds {
            total_prompt_tokens,
            ctx_size: ctx_size - safety_tokens,
        });
    }

    let available_tokens = if let Some(ctx_output_size) = ctx_output_size {
        std::cmp::min(ctx_size - total_prompt_tokens, ctx_output_size) - safety_tokens
    } else {
        ctx_size - total_prompt_tokens - safety_tokens
    };
    if available_tokens == 0 {
        panic!("available_tokens == 0. This should never happen.",);
    }
    Ok(available_tokens - safety_tokens)
}

#[derive(Error, Debug)]
pub enum RequestTokenLimitError {
    #[error("total_prompt_tokens ({total_prompt_tokens}) exceeds ctx_size ({ctx_size})")]
    PromptTokensExceeds {
        total_prompt_tokens: u32,
        ctx_size: u32,
    },

    #[error("GenericPromptError: {e}")]
    GenericPromptError { e: String },
}
