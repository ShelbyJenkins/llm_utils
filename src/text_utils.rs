use regex::Regex;

pub fn filter_drop_words(texts: &[String], drop_words: &[String]) -> Vec<String> {
    texts
        .iter()
        .filter(|s| {
            !s.split_whitespace().any(|w| {
                drop_words
                    .iter()
                    .any(|dw| w.to_lowercase() == dw.to_lowercase())
            })
        })
        .cloned()
        .collect()
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

pub fn clean_text_content(text: &str) -> String {
    let text = convert_white_space(text);
    let text = reduce_excess_whitespace(&text);
    strip_unwanted_chars(&text).trim().to_string()
}

pub fn convert_white_space(text: &str) -> String {
    let patterns = vec![
        (r"\r", "\r"),
        (r"\t", "\t"),
        (r"\n", "\n"),
        (r"\u{00A0}", " "),
        (r"\u{2009}", " "),
        (r"\u{2002}", " "),
        (r"\u{2003}", " "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }
    reduced_text
}

pub fn reduce_excess_whitespace(text: &str) -> String {
    let patterns = vec![
        (r"\r{1,}", "\r"),
        (r"\t{1,}", "\t"),
        // Preserve double newlines for future splitting
        (r"\n{2,}", "\n\n"),
        (r" {1,}", r" "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }
    // Trim leading and trailing spaces
    reduced_text.trim().to_string()
}

pub fn strip_unwanted_chars(text: &str) -> String {
    // Define which chars can be kept; Alpha-numeric chars, punctuation, and whitespaces.
    let allowed_chars = r#"[a-zA-Z0-9.,?!:;'"\-\(\)\[\]\{\}$&@#%^*()\s]"#;

    // Create a regex pattern to match unwanted characters.
    let regex_pattern = format!("[^{}]+", allowed_chars);

    let re = Regex::new(&regex_pattern).unwrap();

    // Remove unwanted chars using regex.
    re.replace_all(text, "").into_owned()
}

pub fn remove_all_white_space_except_space(text: &str) -> String {
    let patterns = vec![
        (r"\r", " "),
        (r"\t", " "),
        (r"\v", " "),
        (r"\f", " "),
        (r"\n", " "),
        // Matches variants of space characters
        (r" ", " "),
        (r"\u00A0", " "),
        (r"\u2009", " "),
        (r"\u2002", " "),
        (r"\u2003", " "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }

    reduce_excess_whitespace(&reduced_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_drop_words() {
        let texts = vec![
            "This is a test sentence.".to_string(),
            "Another example sentence.".to_string(),
            "The color, wow.".to_string(),
        ];
        let drop_words = vec!["is".to_string(), "example".to_string()];
        let expected_result = vec!["The color, wow.".to_string()];
        assert_eq!(filter_drop_words(&texts, &drop_words), expected_result);
    }

    #[test]
    fn test_split_on_newline() {
        let text = "This\nis\na\ntest\nsentence.";
        let expected_result = vec![
            "This".to_string(),
            "is".to_string(),
            "a".to_string(),
            "test".to_string(),
            "sentence.".to_string(),
        ];
        assert_eq!(split_on_newline(text), expected_result);
    }

    #[test]
    fn test_join_with_newline() {
        let texts = vec![
            "This".to_string(),
            "is".to_string(),
            "a".to_string(),
            "test".to_string(),
            "sentence.".to_string(),
        ];
        let expected_result = "This\nis\na\ntest\nsentence.";
        assert_eq!(join_with_newline(&texts), expected_result);
    }

    #[test]
    fn test_split_on_double_newline() {
        let text = "This\n\nis\n\na\n\ntest\n\nsentence.";
        let expected_result = vec![
            "This".to_string(),
            "is".to_string(),
            "a".to_string(),
            "test".to_string(),
            "sentence.".to_string(),
        ];
        assert_eq!(split_on_double_newline(text), expected_result);
    }

    #[test]
    fn test_join_with_double_newline() {
        let texts = vec![
            "This".to_string(),
            "is".to_string(),
            "a".to_string(),
            "test".to_string(),
            "sentence.".to_string(),
        ];
        let expected_result = "This\n\nis\n\na\n\ntest\n\nsentence.";
        assert_eq!(join_with_double_newline(&texts), expected_result);
    }

    #[test]
    fn test_clean_text_content() {
        let text = "   This is a test sentence.   ";
        let expected_result = "This is a test sentence.";
        assert_eq!(clean_text_content(text), expected_result);
    }

    #[test]
    fn test_convert_white_space() {
        let text = "This\tis\na\rtest\u{00A0}sentence.";
        let expected_result = "This\tis\na\rtest sentence.";
        assert_eq!(convert_white_space(text), expected_result);
    }

    #[test]
    fn test_reduce_excess_whitespace() {
        let text = "This   is   a   test   sentence.";
        let expected_result = "This is a test sentence.";
        assert_eq!(reduce_excess_whitespace(text), expected_result);
    }

    #[test]
    fn test_strip_unwanted_chars() {
        let text = r#"This is a "test" sentence. It includes 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*().
        It also includes some advanced punctuation characters that should be removed, such as ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"#;
        let expected_result = r#"This is a "test" sentence. It includes 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*().
        It also includes some advanced punctuation characters that should be removed, such as "#;
        assert_eq!(strip_unwanted_chars(text), expected_result);
    }

    #[test]
    fn test_remove_all_white_space_except_space() {
        let text = "This\tis\na\rtest sentence.";
        let expected_result = "This is a test sentence.";
        assert_eq!(remove_all_white_space_except_space(text), expected_result);
    }
}
