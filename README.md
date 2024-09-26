# llm_utils: Tools for LLMs with minimal abstraction.

`llm_utils` is not a 'framework'. There are no chains, agents, or buzzwords. Abstraction is minimized as much as possible and individual components are easily accessible. For real world examples of how this crate is used, check out the [llm_client crate](https://github.com/ShelbyJenkins/llm_client).


##### Cargo Install

```toml
llm_utils="*"
```


### LocalLlmModel

Everything you need for GGUF models. The `GgugLoader` wraps the loaders for convience. All loaders return a `LocalLlmModel` which contains the tokenizer, metadata, chat template, and anything that can be extract from the GGUF. 


#### GgufPresetLoader

* Presets for popular models like Llama 3, Phi, Mistral/Mixtral, and more
* Loads the best quantized model by calculating the largest quant that will fit in your VRAM

```rust
let model: LocalLlmModel = GgufLoader::default()
    .llama3_1_8b_instruct()
    .preset_with_available_vram_gb(48) // Load the largest quant that will fit in your vram
    .load()?;
```

#### GgufHfLoader

GGUF models from Hugging Face.

```rust
let model: LocalLlmModel = GgufLoader::default()
    .hf_quant_file_url("https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
    .load()?;
```

#### GgufLocalLoader

GGUF models for local storage.

```rust
let model: LocalLlmModel = GgufLoader::default()
    .local_quant_file_path("/root/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/blobs/9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283")
    .load()?;
```

#### ApiLlmModel

* Supports openai, anthropic, perplexity, and adding your own API models
* Supports prompting, tokenization, and price estimation

```rust
    assert_eq!(ApiLlmModel::gpt_4_o(), ApiLlmModel {
        model_id: "gpt-4o".to_string(),
        context_length: 128000,
        cost_per_m_in_tokens: 5.00,
        max_tokens_output: 4096,
        cost_per_m_out_tokens: 15.00,
        tokens_per_message: 3,
        tokens_per_name: 1,
        tokenizer: Arc<LlmTokenizer>,
    })
```

### LlmTokenizer

* Simple abstract API for encoding and decoding allows for abstract LLM consumption across multiple architechtures.
*Hugging Face's Tokenizer library for local models and Tiktoken-rs for OpenAI and Anthropic ([Anthropic doesn't have a publically available tokenizer](https://github.com/javirandor/anthropic-tokenizer).)

```rust
    let tok = LlmTokenizer::new_tiktoken("gpt-4o"); // Get a Tiktoken tokenizer
    let tok = LlmTokenizer::new_from_tokenizer_json("path/to/tokenizer.json"); // From local path
    let tok = LlmTokenizer::new_from_hf_repo(hf_token, "meta-llama/Meta-Llama-3-8B-Instruct"); // From repo
    // From LocalLlmModel or ApiLlmModel
    let tok = model.model_base.tokenizer;
```

### LlmPrompt

* Generate properly formatted prompts for GGUF models, Openai, and Anthropic. Supports chat template strings/tokens and openai hashmaps
* Count prompt tokens and check to ensure it's within model limits
* Uses the GGUF's chat template and Jinja templates to format the prompt to model spec. Build with generation prefixes on all chat template models. Even those that don't explicitly support it.

```rust
// From LocalLlmModel or ApiLlmModel
let prompt: LlmPrompt = LlmPrompt::new_chat_template_prompt(&model);
let prompt: LlmPrompt = LlmPrompt::new_openai_prompt(&model);

// Add system messages
prompt.add_system_message().set_content("You are a nice robot");

// User messages
prompt.add_user_message().set_content("Hello");

// LLM responses
prompt.add_assistant_message().set_content("Well how do you do?");

// Messages all share the same functions see prompting::PromptMessage for more
prompt.add_system_message().append_content(final_rule_set);
prompt.add_system_message().prepend_content(starting_rule_set);

// Builds with generation prefix. The llm will complete the response: 'Don't you think that is... cool?'. 
prompt.set_generation_prefix("Don't you think that is...");

// Get total tokens in prompt
let total_prompt_tokens: u32 = prompt.get_total_prompt_tokens();

// Get chat template formatted prompt
let chat_template_prompt: String = prompt.get_built_prompt_string();
let chat_template_prompt_as_tokens: Vec<u32> = prompt.get_built_prompt_as_tokens()

// Openai formatted prompt (Openai and Anthropic format)
let openai_prompt: Vec<HashMap<String, String>> = prompt.get_built_prompt_hashmap()


// Validate requested max_tokens for a generation. If it exceeds the models limits, reduce max_tokens to a safe value.
let actual_request_tokens = check_and_get_max_tokens(
        model.context_length,
        model.max_tokens_output, // If using a GGUF model use either model.context_length or the ctx_size of the server.
        total_prompt_tokens,
        10,
        requested_max_tokens,
    )?;
```

### Text Processing and NLP 

### TextChunker

Balanced text chunking means that all chunks are approximately the same size. See [my blog post on text chunking for implementation details](https://shelbyjenkins.github.io/blog/text-segmentation-1/).

* A novel *balanced* text chunker that creates chunks of approximately equal length
* More accurate than *unbalanced* implementations that create orphaned final chunks
* Optimized with a parallelization

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

### TextSplitter

* Unicode text segmentation on paragraphs, sentences, words, graphemes
* The *only* semantic sentence segementation implementation in Rust (Please ping me if i'm wrong!) - *mostly* works 

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

#### TextCleaner

* Clean raw text into unicode format
* Reduce duplicate whitespace
* Remove unwanted chars and graphemes

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

#### clean_html

* Clean raw HTML into clean strings of content
* Uses an implementation of Mozilla's [Readability](https://github.com/mozilla/readability) to remove unwanted HTML

#### test_text

* Macro generated test content
* Used for internal testing, but can be used for general LLM test cases

### Grammar Constraints

* Pre-built configurable grammars for fine grained control of open source LLM outputs. Current implementations include booleans, integers, sentences, words, exact strings, and more. Open an issue if you'd like to suggest more
* [Grammars are the most capable method for structuring the output of an LLM.](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) This was designed for use with LlamaCpp, but plan to support others

```rust
// A grammar that constraints to a number between 1 and 4
let integer_grammar: IntegerGrammar = Grammar::integer();
integer_grammar.lower_bound(1).upper_bound(4);
// Sets a stop word to be appended to the end of generation
integer_grammar.set_stop_word_done("Done.");
// Sets the primitive as optional; a stop word can be generated rather than the primitive
integer_grammar.set_stop_word_no_result("None.");

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

### Setter Traits
* All setter traits are public, so you can integrate into your own projects if you wish. 
* For example: `OpenAiModelTrait`,`GgufLoaderTrait`,`AnthropicModelTrait`, and `HfTokenTrait` for loading models 

## Blog Posts

* [Blog post on text chunking](https://shelbyjenkins.github.io/blog/text-segmentation-1/)

### License

This project is licensed under the MIT License.

### Contributing

My motivation for publishing is for someone to point out if I'm doing something wrong!