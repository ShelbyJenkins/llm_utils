# llm_utils: Basic LLM tools, best practices, and minimal abstraction.

`llm_utils` is not a 'framework'. There are no chains, agents, or buzzwords. Abstraction is minimized as much as possible and individual components are easily accessible. Rather than a multitude of [minimal viable implementations](https://shelbyjenkins.github.io/blog/text-segmentation-1/), the focus is on comprehensive, best practice implementations.

For real world examples of how this crate is used, check out the [llm_client crate](https://github.com/ShelbyJenkins/llm_client).

### Model Loading
* `models::open_source_model::OsLlm`
    * `models::open_source_model::preset::LlmPresetLoader`
        * Presets for popular models that includes HF repos for the models, and local copies of tokenizers and chat templates
        * Loads the best quantized model by calculating the largest quant that will fit in your VRAM
        * Supports Llama 3, Phi, Mistral/Mixtral, and more
    * `models::open_source_model::LlmGgufLoader`
        * Loads any GGUF model from HF repo or locally
* `models::api_model::ApiLlm`
    * Supports openai, anthropic, perplexity
    * Supports prompting, tokenization, and price estimation

### Essentials
* `tokenizer::LlmTokenizer`
    * A simple abstraction over HF's tokenizers and tiktoken-rs
    * Load from local or HF.
    * Included in `OsLlm` and `ApiLlm`
* `prompting::LlmPrompt`
    * Build System/User/Assitant prompt messages into formatted prompts
    * Supports chat template strings/tokens and openai hashmaps
    * Count prompt tokens
    * Integrated with `OsLlm` and `ApiLlm`
    * Assemble messages from multiple text inputs
    * Build with generation prefixes on all chat template models. Even those that don't explicitly support it.

### Constraints
* `grammar::Grammar`
    * Pre-built configurable grammars for fine grained control of open source LLM outputs
    * Currently supports Llama.cpp style grammars, but intended to scale to support other grammars in the future.
* `logit_bias`
    * Supports all LLMs that can use logit bias constraints

### Text Processing and NLP 
* `text_utils::chunking::TextChunker`
    * A novel *balanced* text chunker that creates chunks of approximately equal length
    * More accurate than *unbalanced* implementations that create orphaned final chunks
    * Optimized with a parallelization
* `text_utils::splitting::TextSplitter`
    * Unicode text segmentation on paragraphs, sentences, words, graphemes
    * The *only* semantic sentence segementation implementation in Rust (Please ping me if i'm wrong!) - *mostly* works 
* `text_utils::clean_text::TextCleaner`
    * Clean raw text into unicode format
    * Reduce duplicate whitespace
    * Remove unwanted chars and graphemes
* `text_utils::clean_html`
    * Clean raw HTML into clean strings of content
    * Uses an implementation of Mozilla's [Readability](https://github.com/mozilla/readability) to remove unwanted HTML
* `text_utils::test_text`
    * Macro generated test content
    * Used for internal testing, but can be used for general LLM test cases

### Setter Traits
* All setter traits are public, so you can integrate into your own projects if you wish. 
* For example: `models::api_model_openai::OpenAiModelTrait` or `models::open_source_model::hf_loader::HfTokenTrait`

## Installation

```toml
[dependencies]
llm_utils = "*"
```

## Model Loading 🛤️

### LlmPresetLoader

* Presets for Open Source LLMs from Hugging Face, or from local storage
* Load and/or download a model with metadata, tokenizer, and local path (for local LLMs like llama.cpp, vllm, mistral.rs)
* Auto-select the largest quantized GGUF that will fit in your vram!


```rust
    // Load the largest quantized Meta-Llama-3-8B-Instruct model that will fit in your vram
    let model: OsLlm = OsLlmLoader::new()
        .llama3_8b_instruct()
        .vram(48)
        .use_ctx_size(9001) // ctx_size impacts vram usage!
        .load()?;

```

See [example.](examples/model_presets.rs)

### LlmGgufLoader

* GGUF models from Hugging Face or local path 

```rust
    // From HF
    let model_url = "https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q6_K.gguf";
    let model: OsLlm = LlmGgufLoader::new()
        .hf_quant_file_url(model_url)
        .load()?;

    // Note: because we can't instantiate a tokenizer from a GGUF file, the returned model will not have a tokenizer!
    // However, if we provide the base model's repo, we load from there.
    let repo_id = "meta-llama/Meta-Llama-3-8B-Instruct";
    let model: OsLlm = LlmGgufLoader::new()
        .hf_quant_file_url(model_url)
        .hf_config_repo_id(repo_id)
        .load()?;

    // From Local
    let local_path = "/root/.cache/huggingface/hub/models--MaziyarPanahi--Meta-Llama-3-8B-Instruct-GGUF/blobs/c2ca99d853de276fb25a13e369a0db2fd3782eff8d28973404ffa5ffca0b9267";
    let model: OsLlm = LlmGgufLoader::new()
        .local_quant_file_path(local_path)
        .load()?;

    // Again, we require a tokenizer.json. This can also be loaded from a local path.
    let local_config_path = "/llm_utils/src/models/open_source/llama/llama3_8b_instruct";
    let model: OsLlm = LlmGgufLoader::new()
        .local_quant_file_path(model_url)
        .local_config_path(local_config_path)
        .load()?;
```

See [tests for more examples.](src/models/open_source_model/gguf.rs#L150)

### ApiLlm

```rust

    let model: ApiLlm = ApiLlm::gpt_4_o();

    assert_eq!(model, ApiLlm {
        model_id: "gpt-4o".to_string(),
        context_length: 128000,
        cost_per_m_in_tokens: 5.00,
        max_tokens_output: 4096,
        cost_per_m_out_tokens: 15.00,
        tokens_per_message: 3,
        tokens_per_name: 1,
        tokenizer: Arc<LlmTokenizer>,
    })

    // Or Anthropic
    //
    let model: ApiLlm = ApiLlm::claude_3_opus();
```


## Essentials 🧮

### Prompting

- Generate properly formatted prompts for GGUF models, Openai, and Anthropic.

- Uses the GGUF's chat template and Jinja templates to format the prompt to model spec.

```rust
    let model: OsLlm = OsLlmLoader::new().llama3_8b_instruct().load()?;
    let prompt: LlmPrompt = LlmPrompt::new_from_os_llm(&model);
    // or
    let model: ApiLlm = ApiLlm::gpt_4_o();
    let prompt: LlmPrompt = LlmPrompt::new_from_openai_llm(&model);

    // Add system messages
    prompt.add_system_message().set_content("You are a nice robot");

    // User messages
    prompt.add_user_message().set_content("Hello");

    // LLM responses
    prompt.add_assistant_message().set_content("Well how do you do?");

    // Messages all share the same functions see prompting::PromptMessage for more
    prompt.add_system_message().append_content(final_rule_set);
    prompt.add_system_message().prepend_content(starting_rule_set);

    // Build prompt to set the built prompt fields to be sent to the llm
    prompt.build();
    // Build with generation prefix. The llm will complete the response: 'Don't you think that is... cool?'. 
    prompt.build_with_generation_prefix("Don't you think that is...");
    // Build without safety checks (allows to build with assistant as final message) for debug and print
    prompt.build_final();

    // Chat template formatted
    let chat_template_prompt: String = prompt.built_chat_template_prompt.clone()
    let chat_template_prompt_as_tokens: Vec<u32> = prompt.built_prompt_as_tokens.clone()

    // Openai formatted prompt (Openai and Anthropic format)
    let openai_prompt: Vec<HashMap<String, String>> = prompt.built_openai_prompt.clone()

    // Get total tokens in prompt
    let total_prompt_tokens: u32 = prompt.total_prompt_tokens();

    // Validate requested max_tokens for a generation. If it exceeds the models limits, reduce max_tokens to a safe value.
    let actual_request_tokens = check_and_get_max_tokens(
            model.context_length,
            model.max_tokens_output, // If using a GGUF model use either model.context_length or the ctx_size of the server.
            total_prompt_tokens,
            10,
            requested_max_tokens,
        )?;
```

### Tokenizer
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

## Text Processing and NLP 🪓

### Text cleaning

```rust
    // Normalizes all whitespace chars .
    // Reduce the number of newlines to singles or doubles (paragraphs) or convert them to " ".
    // Optionally, remove all characters besides alphabetic, numbers, and punctuation. 
    //
    let mut text_cleaner: String = llm_utils::text_utils::clean_text::TextCleaner::new();
    let cleaned_text: String = text_cleaner
        .reduce_newlines_to_single_space()
        .remove_non_basic_ascii()
        .run(some_dirty_text);

    // Convert HTML to cleaned text.
    // Uses an implementation of Mozilla's readability mode and HTML2Text.
    //
    let cleaned_text: String = llm_utils::text_utils::clean_html::clean_html(raw_html);

```

### Text segmentation

Split text by paragraphs, sentences, words, and graphemes.

```rust
    let paragraph_splits: Vec<String> =  TextSplitter::new()
        .on_two_plus_newline()
        .split_text(&text)?;

    let newline_splits: Vec<String> =  TextSplitter::new()
        .on_single_newline()
        .split_text(&text)?;

    // There is no good implementation sentence splitting in Rust!
    // This implementation is better than unicode-segmentation crate or any other crate I tested.
    // But still not as good as a model based approach like Spacy or other NLP libraries.
    //
    let sentence_splits: Vec<String> =  TextSplitter::new()
        .on_sentences_rule_based()
        .split_text(&text)?;

    // Unicode

    let sentence_splits: Vec<String> =  TextSplitter::new()
        .on_sentences_unicode()
        .split_text(&text)?;

    let word_splits: Vec<String> =  TextSplitter::new()
        .on_words_unicode()
        .split_text(&text)?;

    
    let graphemes_splits: Vec<String> =  TextSplitter::new()
        .on_graphemes_unicode()
        .split_text(&text)?;

    // If the split separator produces less than two splits,
    // this mode tries the next separator.
    // It does this until it produces more than one split.
    //
    let paragraph_splits: Vec<String> =  TextSplitter::new()
        .on_two_plus_newline()
        .recursive(true)
        .split_text(&text)?;
```

### Text chunking 

Balanced text chunking means that all chunks are approximately the same size. 

See [my blog post on text chunking for implementation details](https://shelbyjenkins.github.io/blog/text-segmentation-1/).

```rust
    let text = "one, two, three, four, five, six, seven, eight, nine";

    // Give a max token count of four, other text chunkers would split this into three chunks.
    assert_eq!(["one, two, three, four", "five, six, seven, eight", "nine"], // "nine" is orphaned!
        OtherChunkers::new()
        .max_chunk_token_size(4)
        .Chunk(text));

    // A balanced text chunker, however, would also split the text into three chunks, but of even sizes.
    assert_eq!(["one, two, three", "four, five, six", "seven, eight, nine"], 
        TextChunker::new()
        .max_chunk_token_size(4)
        .run(&text)?);
       
```

As long as the the total token length of the incoming text is not evenly divisible by they max token count, the final chunk will be smaller than the others. In some cases it will be so small it will be "orphaned" and rendered useless. If you asked your RAG implementation `What did seven eat?`, that final chunk that answers the question would not be retrievable. 

The TextChunker first attempts to split semantically in the following order: Paragraphs, newlines, sentences. If that fails it builds chunks linearlly by using the largest available splits, and splitting where needed.

## Constraints #️⃣

### Grammar 
- [Grammars are the most capable method for structuring the output of an LLM.](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) This was designed for use with LlamaCpp, but plan to support others.

- Current implementations include booleans, integers, sentences, words, exact strings, and more. Open an issue if you'd like to suggest more.

```rust
    // A grammar that constraints to a number between 1 and 4
    let integer_grammar: IntegerGrammar = Grammar::integer();
    integer_grammar.lower_bound(1).upper_bound(4);
    // Sets a stop word to be appended to the end of generation
    integer_grammar.set_stop_word_done("Done.");
    // Sets the primitive as optional; a stop word can be generated rather than the primitive
    integer_grammar.set_stop_word_null_result("None.");

    // Returns the string to feed into LLM call
    let grammar_string: String = integer_grammar.grammar_string();

    // Cleans the response and checks if it's valid
    let string_response: Result<String, GrammarError> = integer_grammar.validate_clean(llm_response);
    // Parses the response to the grammar's primitive
    let integer_response: Result<u32, GrammarError>  = integer_grammar.grammar_parse(llm_response);

    // Enum for dynamic abstraction
    let grammar: Grammar = integer_grammar.wrap();
    // The enum implements the same functions that are generic across all grammars
    grammar.set_stop_word_done("Done.");
    let grammar_string: String = grammar.grammar_string();
    let string_response: Result<String, GrammarError> = grammar.validate_clean(llm_response);

```

See [the module for all implemented types](src/grammar/mod.rs#L21)

### Logit bias 
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





### License

This project is licensed under the MIT License.

### Contributing

My motivation for publishing is for someone to point out if I'm doing something wrong!