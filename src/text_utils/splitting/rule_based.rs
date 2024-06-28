use lazy_static::lazy_static;
use regex::Regex;
use std::ops::Range;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

pub fn split_text_into_sentences(text: &str, keep_separator: bool) -> Vec<String> {
    cut(text.to_owned(), keep_separator)
}

pub fn split_text_into_indices(text: &str, keep_separator: bool) -> Vec<Range<usize>> {
    let sentences = cut(text.to_owned(), keep_separator);
    let mut indices: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for sentence in sentences.iter() {
        let end = start + sentence.len();
        indices.push(Range { start, end });
        start = end;
    }
    if !indices.is_empty() {
        let last = indices.last().unwrap();
        if last.end != text.len() {
            eprintln!("split_text_into_indices: indices do not align with input text.");
            return vec![];
        }
    }

    indices
}

// This is taken from https://github.com/indicium-ag/readability-text-cleanup-rs/blob/master/src/katana.rs
// However, the original code is not available in the crate as it does not export it!
fn cut(mut text: String, keep_separator: bool) -> Vec<String> {
    remove_composite_abbreviations(&mut text);
    remove_suspension_points(&mut text);
    remove_floating_point_numbers(&mut text);
    handle_floats_without_leading_zero(&mut text);
    remove_abbreviations(&mut text);
    remove_initials(&mut text);
    unstick_sentences(&mut text);
    remove_sentence_enders_before_parens(&mut text);
    remove_sentence_enders_next_to_quotes(&mut text);
    let sentences = split_sentences(&text);
    let sentences = repair_sentences(sentences);
    if keep_separator {
        sentences
    } else {
        sentences.into_iter().map(|s| s.trim().to_owned()).collect()
    }
}

fn remove_composite_abbreviations(text: &mut String) {
    *text = REMOVE_COMPOSITE_ABBREVIATIONS
        .replace_all(text, "$comp&;&")
        .to_string();
}

fn remove_suspension_points(text: &mut String) {
    *text = REMOVE_SUSPENSION_POINTS
        .replace_all(text, "&&&.")
        .to_string();
}

fn remove_floating_point_numbers(text: &mut String) {
    *text = REMOVE_FLOATING_POINT_NUMBERS
        .replace_all(text, "$number&@&$decimal")
        .to_string();
}

fn handle_floats_without_leading_zero(text: &mut String) {
    *text = HANDLE_FLOATS_WITHOUT_LEADING_ZERO
        .replace_all(text, " &#&$nums")
        .to_string();
}

fn remove_abbreviations(text: &mut String) {
    *text = REMOVE_ABBREVIATIONS
        .replace_all(text, |caps: &regex::Captures| {
            caps.iter()
                .filter_map(|c| c.map(|c| c.as_str().to_string().replace('.', "&-&")))
                .collect::<String>()
        })
        .to_string();
}

fn remove_initials(text: &mut String) {
    *text = REMOVE_INITIALS.replace_all(text, "$init&_&").to_string();
}

fn unstick_sentences(text: &mut String) {
    *text = UNSTICK_SENTENCES
        .replace_all(text, "$left $right")
        .to_string();
}

fn remove_sentence_enders_before_parens(text: &mut String) {
    *text = REMOVE_SENTENCE_ENDERS_BEFORE_PARENS
        .replace_all(text, "&==&$bef")
        .to_string();
}

fn remove_sentence_enders_next_to_quotes(text: &mut String) {
    *text = QUOTE_TRANSFORMATIONS
        .iter()
        .fold(text.to_string(), |acc, (regex, repl)| {
            regex.replace_all(&acc, *repl).to_string()
        });
}

fn is_word_char(c: char) -> bool {
    let group = c.general_category_group();
    group == GeneralCategoryGroup::Letter || group == GeneralCategoryGroup::Number
}

fn is_line_separator_char(c: char) -> bool {
    let group = c.general_category();
    group == GeneralCategory::LineSeparator || group == GeneralCategory::ParagraphSeparator
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut previous_sentence = String::new();

    for c in text.chars() {
        if is_word_char(c) {
            if !previous_sentence.is_empty() {
                sentences.push(previous_sentence);
                previous_sentence = String::new();
            }
            current_sentence.push(c);
        } else if is_line_separator_char(c) {
            if !previous_sentence.is_empty() {
                previous_sentence.push(c);
            } else {
                current_sentence.push(c);
            }
        } else if c == '.' || c == '?' || c == '!' {
            current_sentence.push(c);
            previous_sentence = current_sentence;
            current_sentence = String::new();
        } else if previous_sentence.is_empty() {
            current_sentence.push(c);
        } else {
            previous_sentence.push(c);
        }
    }

    if !previous_sentence.is_empty() {
        sentences.push(previous_sentence);
    }
    if !current_sentence.is_empty() {
        sentences.push(current_sentence);
    }

    sentences
}

fn repair_sentences(sentences: Vec<String>) -> Vec<String> {
    let repaired_sentences: Vec<String> = sentences
        .into_iter()
        .map(|s| {
            let replaced_sentence = s
                .replace("&;&", ".")
                .replace("&&&", "..")
                .replace("&@&", ".")
                .replace("&#&", ".")
                .replace("&-&", ".")
                .replace("&_&", ".")
                .replace("&*&", ".");
            let paren_repaired = PAREN_REPAIR
                .replace_all(&replaced_sentence, r"$1)")
                .to_string();
            QUOTE_REPAIR_REGEXES
                .iter()
                .fold(paren_repaired, |acc, regex| {
                    regex
                        .replace_all(
                            &acc,
                            match regex as *const Regex {
                                x if x == &QUOTE_REPAIR_REGEXES[0] as *const Regex => r#"'$p""#,
                                x if x == &QUOTE_REPAIR_REGEXES[1] as *const Regex => r#"'$p""#,
                                x if x == &QUOTE_REPAIR_REGEXES[2] as *const Regex => r#"$p""#,
                                x if x == &QUOTE_REPAIR_REGEXES[3] as *const Regex => r#"$p""#,
                                x if x == &QUOTE_REPAIR_REGEXES[4] as *const Regex => r#"$p'"#,
                                _ => r#"$p""#,
                            },
                        )
                        .to_string()
                })
        })
        .collect();
    repaired_sentences
}

lazy_static! {
    static ref REMOVE_COMPOSITE_ABBREVIATIONS: Regex =
        Regex::new(r"(?P<comp>et al\.)(?:\.)").unwrap();
    static ref REMOVE_SUSPENSION_POINTS: Regex = Regex::new(r"\.{3}").unwrap();
    static ref REMOVE_FLOATING_POINT_NUMBERS: Regex =
        Regex::new(r"(?P<number>[0-9]+)\.(?P<decimal>[0-9]+)").unwrap();
    static ref HANDLE_FLOATS_WITHOUT_LEADING_ZERO: Regex =
        Regex::new(r"\s\.(?P<nums>[0-9]+)").unwrap();
    static ref REMOVE_ABBREVIATIONS: Regex = Regex::new(r"(?:[A-Za-z]\.){2,}").unwrap();
    static ref REMOVE_INITIALS: Regex = Regex::new(r"(?P<init>[A-Z])(?P<point>\.)").unwrap();
    static ref UNSTICK_SENTENCES: Regex =
        Regex::new(r##"(?P<left>[^.?!]\.|!|\?)(?P<right>[^\s"'])"##).unwrap();
    static ref REMOVE_SENTENCE_ENDERS_BEFORE_PARENS: Regex =
        Regex::new(r##"(?P<bef>[.?!])\s?\)"##).unwrap();
    static ref QUOTE_TRANSFORMATIONS: Vec<(Regex, &'static str)> = vec![
        (
            Regex::new(r##"'(?P<quote>[.?!])\s?""##).unwrap(),
            "&^&$quote"
        ),
        (
            Regex::new(r##"'(?P<quote>[.?!])\s?""##).unwrap(),
            "&**&$quote"
        ),
        (
            Regex::new(r##"(?P<quote>[.?!])\s?""##).unwrap(),
            "&=&$quote"
        ),
        (
            Regex::new(r##"(?P<quote>[.?!])\s?'""##).unwrap(),
            "&,&$quote"
        ),
        (
            Regex::new(r##"(?P<quote>[.?!])\s?'"##).unwrap(),
            "&##&$quote"
        ),
        (Regex::new(r##"(?P<quote>[.?!])\s?""##).unwrap(), "&$quote"),
    ];
    static ref PAREN_REPAIR: Regex = Regex::new(r"&==&(?P<p>[.!?])").unwrap();
    static ref QUOTE_REPAIR_REGEXES: [Regex; 6] = [
        Regex::new(r"&\^&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&\*\*&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&=&(?P<p>[.!?])").unwrap(),
        Regex::new(r#"&,&(?P<p>[.!?])"#).unwrap(),
        Regex::new(r"&##&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&\$&(?P<p>[.!?])").unwrap(),
    ];
}
