use fancy_regex::Regex;
use lazy_static::lazy_static;
use std::borrow::Cow;

#[derive(Default)]
pub enum Newlines {
    Space,
    Single,
    #[default]
    Double,
    None,
}
#[derive(Default)]
pub struct TextCleaner {
    pub newlines: Newlines,
    pub remove_non_basic_ascii: bool,
}
impl TextCleaner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn do_not_reduce_newlines(mut self) -> Self {
        self.newlines = Newlines::None;
        self
    }

    pub fn reduce_newlines_to_single_space(mut self) -> Self {
        self.newlines = Newlines::Space;
        self
    }

    pub fn reduce_newlines_to_single_newline(mut self) -> Self {
        self.newlines = Newlines::Single;
        self
    }

    pub fn reduce_newlines_to_double_newline(mut self) -> Self {
        self.newlines = Newlines::Double;
        self
    }

    pub fn remove_non_basic_ascii(mut self) -> Self {
        self.remove_non_basic_ascii = true;
        self
    }

    pub fn run(&self, text: &str) -> String {
        let mut text = Cow::Borrowed(text);

        text = Cow::Owned(END_OF_LINE_REGEX.replace_all(&text, "\n").into_owned());
        text = Cow::Owned(
            END_OF_PARAGRAPH_REGEX
                .replace_all(&text, "\n\n")
                .into_owned(),
        );
        text = Cow::Owned(WHITE_SPACE_REGEX.replace_all(&text, " ").into_owned());

        text = match self.newlines {
            Newlines::Space => {
                Cow::Owned(SINGLE_NEWLINE_REGEX.replace_all(&text, " ").into_owned())
            }
            Newlines::Single => {
                Cow::Owned(SINGLE_NEWLINE_REGEX.replace_all(&text, "\n").into_owned())
            }
            Newlines::Double => {
                Cow::Owned(DOUBLE_NEWLINE_REGEX.replace_all(&text, "\n\n").into_owned())
            }
            Newlines::None => text,
        };
        if self.remove_non_basic_ascii {
            text = Cow::Owned(UNWANTED_CHARS_REGEX.replace_all(&text, "").into_owned());
        }

        SINGLE_SPACE_REGEX
            .replace_all(&text, " ")
            .trim()
            .to_string()
    }
}

pub fn normalize_whitespace(text: &str) -> String {
    let text = END_OF_LINE_REGEX.replace_all(text, "\n").to_string();
    let text = END_OF_PARAGRAPH_REGEX
        .replace_all(&text, "\n\n")
        .to_string();
    WHITE_SPACE_REGEX.replace_all(&text, " ").to_string()
}

pub fn strip_unwanted_chars(text: &str) -> String {
    UNWANTED_CHARS_REGEX.replace_all(text, "").to_string();
    text.trim().to_string()
}

pub fn reduce_to_single_whitespace(text: &str) -> String {
    let text = SINGLE_SPACE_REGEX.replace_all(text, " ");
    SINGLE_NEWLINE_REGEX
        .replace_all(&text, "\n")
        .trim()
        .to_string()
}

lazy_static! {
    //
    // Newlines
    //
    static ref END_OF_LINE_SEQUENCES: Vec<&'static str> = vec![
        // Ascii
        r"\r\n", // Windows // This must be first to avoid matching \r
        r"\r",   // MacOS
        r"\v",   // Vertical tab
        r"\f",   // Form feed
        // Unicode
        r"\u{2028}",
        ];
    static ref END_OF_LINE_REGEX: Regex = Regex::new(&END_OF_LINE_SEQUENCES.join("|")).unwrap();
    static ref SINGLE_NEWLINE_REGEX: Regex = Regex::new(r" \n{1,}|\n{1,} |\n{1,}").unwrap();
    //
    // Paragraphs
    //
    static ref END_OF_PARAGRAPH_SEQUENCES: Vec<&'static str> = vec![
        // Unicode
        r"\u{2029}",
        ];
    static ref END_OF_PARAGRAPH_REGEX: Regex = Regex::new(&END_OF_PARAGRAPH_SEQUENCES.join("|")).unwrap();
    static ref DOUBLE_NEWLINE_REGEX: Regex = Regex::new(r" \n{2,}|\n{2,} |\n{2,}").unwrap();
    //
    // White space
    //
    static ref WHITE_SPACE_SEQUENCES: Vec<&'static str> = vec![
        // Ascii
        r"\t",
        // Unicode
        r"\u{0020}",
        r"\u{00A0}",
        r"\u{1680}",
        r"\u{2000}",
        r"\u{2001}",
        r"\u{2002}",
        r"\u{2003}",
        r"\u{2004}",
        r"\u{2005}",
        r"\u{2006}",
        r"\u{2007}",
        r"\u{2008}",
        r"\u{2009}",
        r"\u{200A}",
        r"\u{2028}",
        r"\u{202F}",
        r"\u{205F}",
        r"\u{3000}",
        r"\u{0009}",
        ];
    static ref WHITE_SPACE_REGEX: Regex = Regex::new(&WHITE_SPACE_SEQUENCES.join("|")).unwrap();
    static ref SINGLE_SPACE_REGEX: Regex = Regex::new(r" {1,}").unwrap();
    //
    // Unwanted characters
    //
    static ref UNWANTED_CHARS_REGEX: Regex = Regex::new(r#"[^a-zA-Z0-9.,?!:;'\"\-\(\)\[\]\{\}$&@#%^*()\s]+"#).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let ascii_text = "Ascii\tspaces here. Unicode\u{00A0}spaces here.";
        let ascii_result = "Ascii spaces here. Unicode spaces here.";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
        let ascii_text = "Ascii\nnewlines
. Unicode\u{2028}newlines. ";
        let ascii_result = "Ascii\nnewlines\n. Unicode\nnewlines.\n";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
        let ascii_text = "Ascii\n\nparagraphs\r\n\r\n.Unicode\u{2029}paragraphs. ";
        let ascii_result = "Ascii\n\nparagraphs\n\n.Unicode\n\nparagraphs.\n\n";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
    }

    #[test]
    fn test_clean_to_single_spaces() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\n And of course, newlines.\n\n";
        let ascii_result = "Ascii spaces here. Unicode spaces here. And of course, newlines.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_single_space()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_clean_to_single_newlines() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\n And of course, newlines.\n\n Cool.";
        let ascii_result =
            "Ascii spaces here. Unicode spaces here.\nAnd of course, newlines.\nCool.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_single_newline()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_clean_to_double_newlines() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\n\nAscii\n\nparagraphs.\r\n\r\n Unicode\u{2029}paragraphs. Cool.";
        let ascii_result =
            "Ascii spaces here. Unicode spaces here.\n\nAscii\n\nparagraphs.\n\nUnicode\n\nparagraphs.\n\nCool.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_double_newline()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_strip_unwanted_chars() {
        let ascii_text = r#"This is a "test" sentence. It include's 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*(). It also includes some advanced punctuation characters that should be removed, such as ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"#;
        let ascii_result = r#"This is a "test" sentence. It include's 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*(). It also includes some advanced punctuation characters that should be removed, such as"#;
        assert_eq!(
            TextCleaner::new()
                .do_not_reduce_newlines()
                .remove_non_basic_ascii()
                .run(ascii_text),
            ascii_result
        );
    }
}
