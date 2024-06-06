# llm_utils 
Utilities for Llama.cpp, Openai, Anthropic, Mistral-rs. Made for the llm_client crate, but split into it's own crate because some of these are useful!
### Installation
```toml
[dependencies]
llm_utils = "*"

```
### Model presets 🛤️

- Presets for Open Source LLMs from Hugging Face, or API models like OpenAI, and Anthropic.

- Load and/or download a model with metadata, tokenizer, and local path (for local LLMs like llama.cpp, vllm, mistral.rs).

- Auto-select the largest quantized GGUF that will fit in your vram!

Supported Open Source models:

⚪ Llama 3

⚪ Mistral and Mixtral

⚪ Phi 3



```rust
    // Load the largest quantized Mistral-7B-Instruct model that will fit in your vram
    //
    let model: OsLlm = PresetModelBuilder::new()
        .mistral_7b_instruct()
        .vram(48)
        .ctx_size(9001) // ctx_size impacts vram usage!
        .load()
        .await?;

    not_a_real_assert_eq!(model, OsLlm {
        pub model_id: String,
        pub model_url: String,
        pub local_model_path: String, // Use this to load the llama.cpp server
        pub model_config_json: OsLlmConfigJson,
        pub chat_template: OsLlmChatTemplate,
        pub tokenizer: Option<LlmTokenizer>,
    })

    // Or Openai
    //
    let model: OpenAiLlm = OpenAiLlm::gpt_4_o();

    not_a_real_assert_eq!(model, OpenAiLlm {
        model_id: "gpt-4o".to_string(),
        context_length: 128000,
        cost_per_m_in_tokens: 5.00,
        max_tokens_output: 4096,
        cost_per_m_out_tokens: 15.00,
        tokens_per_message: 3,
        tokens_per_name: 1,
        tokenizer: Option<LlmTokenizer>,
    })

    // Or Anthropic
    //
    let model: AnthropicLlm = AnthropicLlm::claude_3_opus();
```

### GGUF models from Hugging Face or local path 🚤

```rust
    // From HF
    //
    let model_url = "https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf";
    let model: OsLlm = GGUFModelBuilder::new()
            .hf_quant_file_url(model_url)
            .load()
            .await?;

    // Note: because we can't instantiate a tokenizer from a GGUF file, the returned model will not have a tokenizer!
    // However, if we provide the base model's repo, we load from there.
    let repo_id = "meta-llama/Meta-Llama-3-8B-Instruct";
    let model: OsLlm = GGUFModelBuilder::new()
        .hf_quant_file_url(model_url)
        .hf_config_repo_id(repo_id)
        .load()
        .await?;

    // From Local
    //
    let local_path = "/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267";
    let model: OsLlm = GGUFModelBuilder::new()
            .local_quant_file_path(local_path)
            .load()
            .await?;

    // Again, we require a tokenizer.json. This can also be loaded from a local path.
    let local_config_path = "/llm_utils/src/models/open_source/llama/llama_3_8b_instruct";
    let model: OsLlm = GGUFModelBuilder::new()
        .local_quant_file_path(model_url)
        .local_config_path(local_config_path)
        .load()
        .await?;
```

### Tokenizer 🧮
- Hugging Face's Tokenizer library for local models and Tiktoken-rs for OpenAI and Anthropic ([Anthropic doesn't have a publically available tokenizer](https://github.com/javirandor/anthropic-tokenizer).)

- Simple abstract API for encoding and decoding allows for abstract LLM consumption across multiple architechtures.

- Safely set the `max_token` param for LLMs to ensure requests don't fail due to exceeding token limits!
```rust
    // Get a Tiktoken tokenizer
    //
    let tokenizer: LlmTokenizer = LlmTokenizer::new_tiktoken("gpt-4o");

    // Get a Hugging Face tokenizer from local path
    //
    let tokenizer: LlmTokenizer = LlmTokenizer::new_from_tokenizer_json("path/to/tokenizer.json");
    
    // Or load from repo
    //
    let tokenizer: LlmTokenizer = LlmTokenizer::new_from_hf_repo(hf_token, "meta-llama/Meta-Llama-3-8B-Instruct");

    // Get tokenizan'
    //
    let token_ids: Vec<u32> = tokenizer.tokenize("Hello there");
    let count: u32 = tokenizer.count_tokens("Hello there");
    let word_probably: String = tokenizer.detokenize_one(token_ids[0])?; 
    let words_probably: String = tokenizer.detokenize_many(token_ids)?; 

    // These function are used for generating logit bias
    let token_id: u32 = tokenizer.try_into_single_token("hello");
    let word_probably: String = tokenizer.try_from_single_token_id(1234);
```

### Prompting 🎶
- Generate properly formatted prompts for GGUF models, Openai, and Anthropic.

- Uses the GGUF's chat template and Jinja templates to format the prompt to model spec.

- Create prompts from a combination of dynamic inputs and/or static inputs from file.
```rust
    // Default formatted prompt (Openai and Anthropic format)
    //
    let default_formatted_prompt: HashMap<String, HashMap<String, String>> = prompting::default_formatted_prompt(
        "You are a nice robot.",
        "path/to/a/file/no_birds_and_bees_yap.yaml",
        "Where do robots come from?"
    )?;

    // Get total tokens in prompt
    //
    let total_prompt_tokens: u32 = model.openai_token_count_of_prompt(&tokenizer, &default_formatted_prompt);


    // Then convert it to be used for a GGUF model
    //
    let gguf_formatted_prompt: String = prompting::convert_default_prompt_to_model_format(
        &default_formatted_prompt,
        &model.chat_template,
    )?;

    // Since the GGUF formatted prompt is just a string, we can just use the generic count_tokens function
    //
    let total_prompt_tokens: u32 = tokenizer.count_tokens(&gguf_formatted_prompt);

    // Validate requested max_tokens for a generation. If it exceeds the models limits, reduce max_tokens to a safe value.
    //
    let safe_max_tokens = get_and_check_max_tokens_for_response(
            model.context_length,
            model.max_tokens_output, // If using a GGUF model use either model.context_length or the ctx_size of the server.
            total_prompt_tokens,
            10,
            None,
            requested_max_tokens,
        )?;
```

### Grammar 🤓
- [Grammars are the most capable method for structuring the output of an LLM.](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) This was designed for use with LlamaCpp, but plan to support others.

- Create lists of N items, restrict character types.

- More to be added (JSON, classification, restrict characters, words, phrases)
```rust
    // Return a list of between 1, 4 items
    //
    let grammar = llm_utils::grammar::create_list_grammar(1, 4);

    // List will be formatted: `- <list text>\n
    //
    let response: String = text_generation_request(&req_config, Some(&grammar)).await?;

    // So you can easily split like:
    //
    let response_items: Vec<String> = response
        .lines()
        .map(|line| line[1..].trim().to_string())
        .collect();

    // Exclude numbers from text generation
    //
    let grammar = llm_utils::grammar::create_text_structured_grammar(vec![RestrictedCharacterSet::PunctuationExtended]);
    let response: String = text_generation_request(&req_config, Some(&grammar)).await?;
    assert!(!response.contains('0'))
    assert!(!response.contains("1234"))

    // Exclude a list of common, and commonly unwanted characters from text generation
    //
    let grammar = llm_utils::grammar::create_text_structured_grammar(vec![RestrictedCharacterSet::PunctuationExtended]);
    let response: String = text_generation_request(&req_config, Some(&grammar)).await?;
    assert!(!response.contains('@'))
    assert!(!response.contains('['))
    assert!(!response.contains('*'))
```

### Logit bias #️⃣
- Create properly formatted logit bias requests for LlamaCpp and Openai.

- Functionality to add logit bias from a variety of sources, along with validation.
```rust
    // Exclude some tokens from text generation
    //
    let mut words = HashMap::new();
    words.entry("delve").or_insert(-100.0);
    words.entry("as an ai model").or_insert(-100.0);

    // Build and validate
    //
    let logit_bias = logit_bias::logit_bias_from_words(&tokenizer, &words)
    let validated_logit_bias = logit_bias::validate_logit_bias_values(&logit_bias)?;

    // Convert
    //
    let openai_logit_bias = logit_bias::convert_logit_bias_to_openai_format(&validated_logit_bias)?;
    let llama_logit_bias = logit_bias::convert_logit_bias_to_llama_format(&validated_logit_bias)?;
```

### Text utils 📝
- Generic utils for cleaning text. Mostly useful for RAG.

- Will add text splitting in the future.


### License

This project is licensed under the MIT License.

### Contributing

My motivation for publishing is for someone to point out if I'm doing something wrong!