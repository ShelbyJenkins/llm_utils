# llm_utils 
Utilities for Llama.cpp, Openai, Anthropic, Mistral-rs. Made for the llm_client crate, but split into it's own crate because some of these are useful!
### Installation
```toml
[dependencies]
llm_utils = "*"
```
### Model loading 🛤️
- Presets for popular GGUF models, along with pre-populated models for Openai and Anthropic.

- Load GGUF models from Hugging Face with auto-quantization level picking based on vram.
```rust
    // Download the largest quantized Mistral-7B-Instruct model that will fit in your vram
    //
    let model: GGUFModel = GGUFModelBuilder::default()
        .mistral_7b_instruct()
        .vram(48)
        .ctx_size(9001) // ctx_size impacts vram usage!
        .load()
        .await?;

    // Or just load directly from a url
    //
    let model: GGUFModel = GGUFModelBuilder::new(hf_token.clone())
            .from_quant_file_url(quantized_model_url)
            .load()
            .await?;

    not_a_real_assert_eq!(model, GGUFModel {
        pub model_id: String,
        pub local_model_path: String, // Use this to load the llama.cpp server
        pub metadata: GGUFMetadata,
    })

    not_a_real_assert_eq!(model.metadata, GGUFMetadata {
        pub embedding_length: u32,    // hidden_size
        pub head_count: u32,          // num_attention_heads
        pub feed_forward_length: u32, // intermediate_size
        pub context_length: u32,      // max_position_embeddings
        pub chat_template: String,    // tokenizer.chat_template
    })

    // Or Openai
    //
    let model: OpenAiModel = OpenAiModel::gpt_4_o();

    let model: OpenAiModel = OpenAiModel::openai_backend_from_model_id("gpt-4o");

    not_a_real_assert_eq!(model, OpenAiModel {
        model_id: "gpt-4o".to_string(),
        context_length: 128000,
        cost_per_m_in_tokens: 5.00,
        max_tokens_output: 4096,
        cost_per_m_out_tokens: 15.00,
        tokens_per_message: 3,
        tokens_per_name: 1,
    })
```

### Tokenizer 🧮
- Hugging Face's Tokenizer library for local models and Tiktoken-rs for Openai.

- Simple abstract API for encoding and decoding allows for abstract LLM consumption across multiple architechtures.

- Safely set the `max_token` param for LLMs to ensure requests don't fail due to exceeding token limits!
```rust
    // Get a tokenizer
    //
    let tokenizer: LlmUtilsTokenizer = LlmUtilsTokenizer::new_tiktoken("gpt-4o");

    // Or from hugging face... 
    // need to add support for a tokenizer from GGUF, but this does not exist yet. 
    // So the tokenizer does not work unless you can first load the tokenizer.json from the original repo
    // as the GGUF format doesn't not include it.
    //
    let tokenizer: LlmUtilsTokenizer = LlmUtilsTokenizer::new_from_model(tokenizers::Tokenizer::from_file("path/to/tokenizer.json"));

    let token_ids: Vec<u32> = tokenizer.tokenize("Hello there");

    // This function is used for generating logit bias
    let token_id: u32 = tokenizer.try_into_single_token("hello");
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
        "path/to/a/file.yaml",
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