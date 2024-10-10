# llm_utils: The best possible text chunker and text splitter and other text tools

Note: Many of the tools that were previously in this crate have been split into sub-crates of the [llm_client](https://github.com/ShelbyJenkins/llm_client) project.


##### Cargo Install

```toml
llm_utils="*"
```


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


## Blog Posts

* [Blog post on text chunking](https://shelbyjenkins.github.io/blog/text-segmentation-1/)

### License

This project is licensed under the MIT License.

### Contributing

My motivation for publishing is for someone to point out if I'm doing something wrong!