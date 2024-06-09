use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

// This is taken from https://github.com/indicium-ag/readability-text-cleanup-rs/blob/master/src/katana.rs
// However, the original code is not available in the crate as it desn't not export it!
pub fn split_text_into_sentences_regex(text: &str) -> Vec<String> {
    cut(text)
}

pub fn split_text_into_sentences_unicode(text: &str) -> Vec<String> {
    text.unicode_sentences().map(|s| s.to_string()).collect()
}

pub fn split_text_into_word_indices_ranges_unicode(text: &str) -> Vec<String> {
    let text = super::clean_text::TextCleaner::new()
        .reduce_newlines_to_single_space()
        .run(text);

    let mut words = Vec::new();
    let indices: Vec<(usize, &str)> = text.unicode_word_indices().collect();
    for i in 0..indices.len() {
        let start_index = indices[i].0;
        let end_index = if i + 1 < indices.len() {
            indices[i + 1].0
        } else {
            text.len()
        };
        let word = &text[start_index..end_index];
        words.push(word.to_string());
    }

    words
}

pub fn split_text_into_graphemes_unicode(text: &str) -> Vec<String> {
    text.graphemes(true).map(|s: &str| s.to_string()).collect()
}

pub fn split_on_newline(text: &str) -> Vec<String> {
    text.split('\n').map(|s| s.to_string()).collect()
}

pub fn join_with_newline(texts: &[String]) -> String {
    texts.join("\n").trim().to_string()
}

pub fn split_on_double_newline(text: &str) -> Vec<String> {
    text.split("\n\n").map(|s| s.to_string()).collect()
}

pub fn join_with_double_newline(texts: &[String]) -> String {
    texts.join("\n\n").trim().to_string()
}

fn remove_composite_abbreviations(text: &str) -> String {
    Regex::new(r"(?P<comp>et al\.)(?:\.)")
        .unwrap()
        .replace_all(text, "$comp&;&")
        .to_string()
}

fn remove_suspension_points(text: &str) -> String {
    Regex::new(r"\.{3}")
        .unwrap()
        .replace_all(text, "&&&.")
        .to_string()
}

fn remove_floating_point_numbers(text: &str) -> String {
    Regex::new(r"(?P<number>[0-9]+)\.(?P<decimal>[0-9]+)")
        .unwrap()
        .replace_all(text, "$number&@&$decimal")
        .to_string()
}

fn handle_floats_without_leading_zero(text: &str) -> String {
    Regex::new(r"\s\.(?P<nums>[0-9]+)")
        .unwrap()
        .replace_all(text, " &#&$nums")
        .to_string()
}

fn remove_abbreviations(text: &str) -> String {
    Regex::new(r"(?:[A-Za-z]\.){2,}")
        .unwrap()
        .replace_all(text, |caps: &regex::Captures| {
            caps.iter()
                .filter_map(|c| c.map(|c| c.as_str().to_string().replace('.', "&-&")))
                .collect::<String>()
        })
        .to_string()
}

fn remove_initials(text: &str) -> String {
    Regex::new(r"(?P<init>[A-Z])(?P<point>\.)")
        .unwrap()
        .replace_all(text, "$init&_&")
        .to_string()
}

fn remove_titles(text: &str) -> String {
    Regex::new(r"(?P<title>[A-Z][a-z]{1,3})(\.)")
        .unwrap()
        .replace_all(text, "$title&*&")
        .to_string()
}

fn unstick_sentences(text: &str) -> String {
    Regex::new(r##"(?P<left>[^.?!]\.|!|\?)(?P<right>[^\s"'])"##)
        .unwrap()
        .replace_all(text, "$left $right")
        .to_string()
}

fn remove_sentence_enders_before_parens(text: &str) -> String {
    Regex::new(r##"(?P<bef>[.?!])\s?\)"##)
        .unwrap()
        .replace_all(text, "&==&$bef")
        .to_string()
}

fn remove_sentence_enders_next_to_quotes(text: &str) -> String {
    let transformations = [
        (r##"'(?P<quote>[.?!])\s?""##, "&^&$quote"),
        (r##"'(?P<quote>[.?!])\s?”"##, "&**&$quote"),
        (r##"(?P<quote>[.?!])\s?”"##, "&=&$quote"),
        (r##"(?P<quote>[.?!])\s?'""##, "&,&$quote"),
        (r##"(?P<quote>[.?!])\s?'"##, "&##&$quote"),
        (r##"(?P<quote>[.?!])\s?""##, "&$quote"),
    ];
    transformations
        .iter()
        .fold(text.to_string(), |acc, (pattern, repl)| {
            Regex::new(pattern)
                .unwrap()
                .replace_all(&acc, *repl)
                .to_string()
        })
}

fn split_sentences(text: &str) -> Vec<Vec<String>> {
    let mut paragraphs: Vec<Vec<String>> = Vec::new();
    let mut current_sentence = String::new();
    let mut current_paragraph = Vec::new();

    for c in text.chars() {
        if c == '\n' {
            if !current_sentence.is_empty() {
                current_paragraph.push(current_sentence.clone());
                current_sentence.clear();
            }
            if !current_paragraph.is_empty() {
                paragraphs.push(current_paragraph.clone());
                current_paragraph.clear();
            }
        } else {
            current_sentence.push(c);
            if c == '.' || c == '?' || c == '!' {
                current_paragraph.push(current_sentence.clone());
                current_sentence.clear();
            }
        }
    }

    if !current_sentence.is_empty() {
        current_paragraph.push(current_sentence);
    }
    if !current_paragraph.is_empty() {
        paragraphs.push(current_paragraph);
    }

    paragraphs
}

fn repair_sentences(paragraphs: Vec<Vec<String>>) -> Vec<String> {
    let paren_repair = Regex::new(r"&==&(?P<p>[.!?])").unwrap();
    let quote_repair_regexes = [
        Regex::new(r"&\^&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&\*\*&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&=&(?P<p>[.!?])").unwrap(),
        Regex::new(r#"&,&(?P<p>[.!?])"#).unwrap(),
        Regex::new(r"&##&(?P<p>[.!?])").unwrap(),
        Regex::new(r"&\$&(?P<p>[.!?])").unwrap(),
    ];

    let repaired_paragraphs: Vec<Vec<String>> = paragraphs
        .into_iter()
        .map(|paragraph| {
            paragraph
                .into_iter()
                .map(|s| {
                    let replaced_sentence = s
                        .trim()
                        .replace("&;&", ".")
                        .replace("&&&", "..")
                        .replace("&@&", ".")
                        .replace("&#&", ".")
                        .replace("&-&", ".")
                        .replace("&_&", ".")
                        .replace("&*&", ".");
                    let paren_repaired = paren_repair
                        .replace_all(&replaced_sentence, r"$1)")
                        .to_string();
                    quote_repair_regexes
                        .iter()
                        .fold(paren_repaired, |acc, regex| {
                            regex
                                .replace_all(
                                    &acc,
                                    match regex as *const Regex {
                                        x if x == &quote_repair_regexes[0] as *const Regex => {
                                            r#"'$p""#
                                        }
                                        x if x == &quote_repair_regexes[1] as *const Regex => {
                                            r#"'$p”"#
                                        }
                                        x if x == &quote_repair_regexes[2] as *const Regex => {
                                            r#"$p”"#
                                        }
                                        x if x == &quote_repair_regexes[3] as *const Regex => {
                                            r#"$p""#
                                        }
                                        x if x == &quote_repair_regexes[4] as *const Regex => {
                                            r#"$p'"#
                                        }
                                        _ => r#"$p""#,
                                    },
                                )
                                .to_string()
                        })
                })
                .filter(|s| !s.is_empty())
                .collect()
        })
        .filter(|p: &Vec<String>| !p.is_empty())
        .collect();

    repaired_paragraphs.into_iter().flatten().collect()
}

fn cut(origin_text: &str) -> Vec<String> {
    let mut text = super::clean_text::TextCleaner::new()
        .reduce_newlines_to_single_space()
        .run(origin_text);
    text = remove_composite_abbreviations(&text);
    text = remove_suspension_points(&text);
    text = remove_floating_point_numbers(&text);
    text = handle_floats_without_leading_zero(&text);
    text = remove_abbreviations(&text);
    text = remove_initials(&text);
    text = remove_titles(&text);
    text = unstick_sentences(&text);
    text = remove_sentence_enders_before_parens(&text);
    text = remove_sentence_enders_next_to_quotes(&text);
    let paragraphs = split_sentences(&text);
    repair_sentences(paragraphs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_utils::test_text::*;
    // fn eval_splits(test_splits: Vec<String>, correct_splits: &[String]) {
    //     for (test, correct) in test_splits.iter().zip(correct_splits.iter()) {
    //         assert_eq!(test, correct);
    //     }
    // }

    fn assert_splits(test_splits: Vec<String>) {
        assert!(!test_splits.is_empty());
        for (i, test) in test_splits.iter().enumerate() {
            println!("{}: {}", i, test);
        }
    }

    // #[test]
    // fn test_sentence_segmentation() {
    //     eval_splits(
    //         split_text_into_sentences_regex(&TURING.test_content),
    //         &TURING.sentence_splits,
    //     );
    //     eval_splits(
    //         split_text_into_sentences_regex(&KATANA.test_content),
    //         &KATANA.sentence_splits,
    //     );
    //     eval_splits(
    //         split_text_into_sentences_regex(&SHAKE.test_content),
    //         &SHAKE.sentence_splits,
    //     );
    // }

    #[test]
    fn test_unicode_sentence_segmentation() {
        assert_splits(split_text_into_sentences_unicode(&TURING.test_content));
        assert_splits(split_text_into_sentences_unicode(&KATANA.test_content));
        assert_splits(split_text_into_sentences_unicode(&SHAKE.test_content));
    }

    #[test]
    fn test_unicode_word_segmentation() {
        assert_splits(split_text_into_word_indices_ranges_unicode(
            &TURING.test_content,
        ));
        assert_splits(split_text_into_word_indices_ranges_unicode(
            &KATANA.test_content,
        ));
        assert_splits(split_text_into_word_indices_ranges_unicode(
            &SHAKE.test_content,
        ));
    }
}
