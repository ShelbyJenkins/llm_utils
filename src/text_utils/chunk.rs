use super::clean_text::TextCleaner;
use lazy_static::lazy_static;
use regex::Regex;

pub fn chunk_text(text: &str, max_length: usize, overlap_percent: Option<usize>) -> Vec<String> {
    let tokenizer = crate::tokenizer::LlmTokenizer::new_tiktoken("gpt-4");
    let mut splitter = DFSTextSplitter::new(max_length, overlap_percent, &tokenizer);
    splitter.run(text).unwrap()
}

lazy_static! {
    static ref SINGLE_NEWLINE_REGEX: Regex = Regex::new(r"\n").unwrap();
    static ref DOUBLE_NEWLINE_REGEX: Regex = Regex::new(r"\n\n").unwrap();
    static ref WHITE_SPACE_REGEX: Regex = Regex::new(r" ").unwrap();
}

#[derive(Clone)]
enum Separator {
    MultiEol,
    SingleEol,
    Sentences,
    Words,
    Whitespace,
    Graphemes,
}

impl Separator {
    fn get_all() -> Vec<Self> {
        vec![
            Self::MultiEol,
            Self::SingleEol,
            Self::Sentences,
            Self::Words,
            Self::Whitespace,
            Self::Graphemes,
        ]
    }
    fn split(&self, text: &str) -> Vec<String> {
        match self {
            Self::MultiEol => {
                let splits = Self::split_text_with_regex(text, &SINGLE_NEWLINE_REGEX, false);
                splits.iter().map(|s| format!("{}{}", s, "\n\n")).collect()
            }
            Self::SingleEol => {
                let splits = Self::split_text_with_regex(text, &DOUBLE_NEWLINE_REGEX, true);
                splits.iter().map(|s| format!("{}{}", s, "\n")).collect()
            }
            Self::Sentences => {
                let splits = crate::text_utils::split::split_text_into_sentences_unicode(text);
                splits.iter().map(|s| format!("{} ", s)).collect()
            }
            Self::Words => {
                crate::text_utils::split::split_text_into_word_indices_ranges_unicode(text)
            }
            Self::Whitespace => Self::split_text_with_regex(text, &WHITE_SPACE_REGEX, true),
            Self::Graphemes => crate::text_utils::split::split_text_into_graphemes_unicode(text),
        }
    }
    fn split_text_with_regex(text: &str, pattern: &Regex, keep_separator: bool) -> Vec<String> {
        if keep_separator {
            // We use captures_iter to include the separators in the output.
            pattern
                .captures_iter(text)
                .fold(Vec::new(), |mut acc, cap| {
                    // Get the text before the match (split part)
                    let start = cap.get(0).map_or(0, |m| m.start());
                    if let Some(last_match) = acc.last() {
                        let last_end = last_match.as_ptr() as usize + last_match.len();
                        let text_start = text.as_ptr() as usize;
                        if last_end > text_start && last_end - text_start < start {
                            // Push the segment before the separator if there is any
                            acc.push(text[last_end - text_start..start].to_string());
                        }
                    }
                    // Push the matched separator
                    acc.push(cap.get(0).unwrap().as_str().to_string());
                    acc
                })
        } else {
            // Simple split, removing empty entries
            pattern
                .split(text)
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
    }
}

/// Splits text that attempts to split by paragraph, newlines, sentences, spaces, and finally chars.
/// After splitting, creating the chunks is used with a DFS algo utilizing memoization and a heuristic prefilter.
pub struct DFSTextSplitter<'a> {
    max_length: usize,
    goal_length_max_threshold: usize,
    goal_length_min_threshold: usize,
    chunk_overlap_max_threshold: usize,
    chunk_overlap_min_threshold: usize,
    chunk_overlap: usize,
    average_range_min: usize,
    memo: std::collections::HashMap<usize, Vec<usize>>,
    original_goal_length: usize,
    goal_length: usize,
    overlap_percent: Option<usize>,
    separators: Vec<Separator>,
    threshold_modifier: f32,
    min_length: f32,

    tokenizer: &'a crate::tokenizer::LlmTokenizer,
}

impl<'a> DFSTextSplitter<'a> {
    pub fn new(
        goal_length: usize,
        overlap_percent: Option<usize>,
        tokenizer: &'a crate::tokenizer::LlmTokenizer,
    ) -> Self {
        let overlap_percent = if let Some(overlap_percent) = overlap_percent {
            if !(10..=100).contains(&overlap_percent) {
                Some(10)
            } else {
                Some(overlap_percent)
            }
        } else {
            Some(10)
        };
        Self {
            max_length: 0,
            goal_length_max_threshold: 0,
            goal_length_min_threshold: 0,
            chunk_overlap_max_threshold: 0,
            chunk_overlap_min_threshold: 0,
            chunk_overlap: 0,
            average_range_min: 0,
            memo: std::collections::HashMap::new(),
            original_goal_length: goal_length,
            goal_length,
            overlap_percent,
            separators: Separator::get_all(),
            threshold_modifier: 0.1,
            min_length: 0.7,

            tokenizer,
        }
    }

    pub fn run(&mut self, text: &str) -> Option<Vec<String>> {
        let text = TextCleaner::new()
            .reduce_newlines_to_double_newline()
            .run(text);

        self.set_thresholds(Some(self.original_goal_length));

        // Skip if too small
        if (self.tokenizer.count_tokens(&text)).lt(&(self.max_length as u32)) {
            return Some(vec![text.to_owned()]);
        }

        for separator in self.separators.clone() {
            self.set_thresholds(Some(self.original_goal_length));
            let splits = separator.split(&text);
            if splits.is_empty() {
                continue;
            }

            while (self.goal_length as f32 / self.original_goal_length as f32) > self.min_length {
                self.memo.clear();

                if self.set_heuristics(&text, &splits) {
                    if let Some(chunk_end_splits) = self.find_valid_chunk_combinations(&splits) {
                        let text_chunks = self.create_chunks(chunk_end_splits, &splits);
                        if text_chunks.is_some() {
                            return text_chunks;
                        }
                    }
                }

                self.set_thresholds(None);
            }
        }

        None
    }

    fn set_thresholds(&mut self, goal_length: Option<usize>) {
        if let Some(length) = goal_length {
            self.goal_length = length;
        } else {
            self.goal_length = self.goal_length - (0.02 * self.goal_length as f32) as usize;
        }

        self.max_length = (self.goal_length as f32 * 1.25) as usize;
        self.goal_length_max_threshold =
            self.goal_length + (self.threshold_modifier * self.goal_length as f32) as usize;
        self.goal_length_min_threshold =
            self.goal_length - (self.threshold_modifier * self.goal_length as f32) as usize;
        if let Some(overlap_percent) = self.overlap_percent {
            self.chunk_overlap =
                (self.goal_length as f32 * (overlap_percent as f32 / 100.0)) as usize;
            self.chunk_overlap_max_threshold =
                self.chunk_overlap + (self.threshold_modifier * self.chunk_overlap as f32) as usize;
            self.chunk_overlap_min_threshold =
                self.chunk_overlap - (self.threshold_modifier * self.chunk_overlap as f32) as usize;
        }
    }

    /// Sets some values that we use as a pre-filter to speed up the process.
    fn set_heuristics(&mut self, text: &str, splits: &[String]) -> bool {
        self.average_range_min = 0;

        for split in splits {
            if (self.tokenizer.count_tokens(split)).gt(&(self.max_length as u32)) {
                return false;
            }
        }

        let total_tokens = self.tokenizer.count_tokens(text);
        let mut estimated_chunks = total_tokens / self.goal_length as u32;
        if estimated_chunks == 1 {
            estimated_chunks = 2;
        }

        let estimated_splits_per_chunk = splits.len() / estimated_chunks as usize;

        // Test required chunks exceed splits
        if splits.len() < estimated_splits_per_chunk {
            return false;
        }

        self.average_range_min = estimated_splits_per_chunk / 2;
        true
    }

    /// Initializes the chunk combo finding process.
    fn find_valid_chunk_combinations(&mut self, splits: &[String]) -> Option<Vec<usize>> {
        let chunks_as_splits = self.recursive_chunk_tester(0, splits);

        if chunks_as_splits.is_none() || chunks_as_splits.as_ref().unwrap().len() < 2 {
            None
        } else {
            chunks_as_splits
        }
    }

    /// Manages the testing of chunk combos.
    /// Stops when it successfuly finds a path to the final split.
    fn recursive_chunk_tester(&mut self, start: usize, splits: &[String]) -> Option<Vec<usize>> {
        let valid_ends = match self.find_valid_endsplits_for_chunk(start, splits) {
            Some(valid_ends) => valid_ends,
            None => return None,
        };

        let mut valid_ends_filtered = Vec::new();
        let mut found_exit_condition = false;

        for &end_split in &valid_ends {
            // Successful exit condition
            if end_split + 1 == splits.len() {
                found_exit_condition = true;
                break;
            }
            // This keeps things from melting
            if end_split != start {
                valid_ends_filtered.push(end_split);
            }
        }

        if found_exit_condition {
            return Some(vec![valid_ends[valid_ends.len() - 1]]);
        }

        for &end_split in &valid_ends_filtered {
            // Recursive call with the next start
            let next_chunk = self.recursive_chunk_tester(end_split, splits);
            // If a valid combination was found in the recursive call
            if next_chunk.is_some() {
                let mut result = vec![end_split];
                result.extend(next_chunk.unwrap());
                return Some(result);
            }
        }

        // If no valid chunking found for the current start, return None
        None
    }

    /// Returns endsplits that are within the threshold of goal_length.
    /// Uses memoization to save from having to recompute.
    /// Starts calculation at + self.average_range_min as a pre-filter.
    fn find_valid_endsplits_for_chunk(
        &mut self,
        start: usize,
        splits: &[String],
    ) -> Option<Vec<usize>> {
        if let Some(valid_ends) = self.memo.get(&{ start }) {
            return Some(valid_ends.clone());
        }

        let mut valid_ends = Vec::new();

        for j in start + 1 + self.average_range_min..splits.len() {
            // Final tokenization will be of combined chunks - not individual chars!
            let current_length = self.tokenizer.count_tokens(&splits[start..j].join("")) as usize;

            if self.overlap_percent.is_some() {
                if current_length
                    >= self.goal_length_min_threshold - self.chunk_overlap_max_threshold
                {
                    if current_length <= self.max_length - self.chunk_overlap_max_threshold {
                        valid_ends.push(j);
                    } else {
                        break;
                    }
                }
            } else if current_length <= self.max_length {
                valid_ends.push(j);
            } else {
                break;
            }
        }

        self.memo.insert(start, valid_ends.clone());

        if valid_ends.is_empty() {
            None
        } else {
            Some(valid_ends)
        }
    }

    /// Creates the text chunks including overlap.
    fn create_chunks(
        &self,
        mut chunk_split_indexes: Vec<usize>,
        splits: &[String],
    ) -> Option<Vec<String>> {
        let mut chunks = Vec::new();

        // So we start at the first split
        if chunk_split_indexes[0] != 0 {
            chunk_split_indexes.insert(0, 0);
        }

        for (i, &split_index) in chunk_split_indexes.iter().enumerate() {
            if i + 1 == chunk_split_indexes.len() {
                break;
            }
            let chunk_splits = &splits[split_index..chunk_split_indexes[i + 1] + 1];
            let text_chunk = chunk_splits.join(" ");
            if text_chunk.is_empty() {
                return None;
            }
            let text_chunk: String = if self.overlap_percent.is_some() {
                if split_index == 0 {
                    let forward_overlap_text = self.create_forward_overlap(
                        chunk_split_indexes[i + 1],
                        chunk_split_indexes[i + 2],
                        self.chunk_overlap_min_threshold,
                        self.chunk_overlap_max_threshold,
                        splits,
                    );

                    if let Some(forward_overlap_text) = forward_overlap_text {
                        format!("{}{}", text_chunk, forward_overlap_text)
                    } else {
                        return None;
                    }
                } else if chunk_split_indexes[i + 1] + 1 == splits.len() {
                    let backwards_overlap_text = self.create_backwards_overlap(
                        split_index,
                        chunk_split_indexes[i - 1],
                        self.chunk_overlap_min_threshold,
                        self.chunk_overlap_max_threshold,
                        splits,
                    );

                    if let Some(backwards_overlap_text) = backwards_overlap_text {
                        format!("{}{}", backwards_overlap_text, text_chunk)
                    } else {
                        return None;
                    }
                } else {
                    let forward_overlap_text = self.create_forward_overlap(
                        chunk_split_indexes[i + 1],
                        chunk_split_indexes[i + 2],
                        self.chunk_overlap_min_threshold / 2,
                        self.chunk_overlap_max_threshold / 2,
                        splits,
                    );
                    let backwards_overlap_text = self.create_backwards_overlap(
                        split_index,
                        chunk_split_indexes[i - 1],
                        self.chunk_overlap_min_threshold / 2,
                        self.chunk_overlap_max_threshold / 2,
                        splits,
                    );
                    if forward_overlap_text.is_none() || backwards_overlap_text.is_none() {
                        return None;
                    } else {
                        format!(
                            "{}{}{}",
                            backwards_overlap_text.unwrap(),
                            text_chunk,
                            forward_overlap_text.unwrap()
                        )
                    }
                }
            } else {
                text_chunk
            };
            let text_chunk = super::clean_text::reduce_to_single_whitespace(&text_chunk);
            let token_count = self.tokenizer.count_tokens(&text_chunk) as usize;

            if token_count > self.max_length {
                return None;
            }

            chunks.push(text_chunk);
        }

        Some(chunks)
    }

    /// Creates forward overlap chunks.
    fn create_forward_overlap(
        &self,
        end_split: usize,
        next_start: usize,
        overlap_min: usize,
        overlap_max: usize,
        splits: &[String],
    ) -> Option<String> {
        let overlap_text = splits[end_split + 1..next_start].join(" ");

        for separator in &self.separators {
            let overlap_splits = separator.split(&overlap_text);
            if overlap_splits.is_empty() {
                continue;
            }
            let mut saved_splits = Vec::new();

            for split in overlap_splits {
                saved_splits.push(split);
                let current_split = saved_splits.join("");
                let cleaned_split = super::clean_text::reduce_to_single_whitespace(&current_split);
                let current_length = self.tokenizer.count_tokens(&cleaned_split) as usize;

                if current_length > overlap_max {
                    break;
                }

                if current_length >= overlap_min {
                    return Some(cleaned_split);
                }
            }
        }

        None
    }

    /// Creates backwards overlap chunks.
    fn create_backwards_overlap(
        &self,
        start_split: usize,
        previous_end: usize,
        overlap_min: usize,
        overlap_max: usize,
        splits: &[String],
    ) -> Option<String> {
        let overlap_text = splits[previous_end..start_split].join(" ");

        for separator in &self.separators {
            let overlap_splits = separator.split(&overlap_text);
            if overlap_splits.is_empty() {
                continue;
            }
            let mut saved_splits = Vec::new();

            for j in (0..overlap_splits.len()).rev() {
                saved_splits.insert(0, &overlap_splits[j]);
                let current_split = saved_splits
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>()
                    .join("");
                let cleaned_split = super::clean_text::reduce_to_single_whitespace(&current_split);
                let current_length = self.tokenizer.count_tokens(&cleaned_split) as usize;

                if current_length > overlap_max {
                    break;
                }

                if current_length >= overlap_min {
                    return Some(current_split);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_utils::test_text::*;

    #[test]
    fn test() {
        let start_time = std::time::Instant::now();
        let res = chunk_text(&LONG.test_content, 512, Some(10));
        let elapsed = start_time.elapsed();
        println!("{:?}", res);
        println!("{:?}", res.len());
        println!("{:?}", elapsed);
    }
    #[test]
    fn test_long() {
        let start_time = std::time::Instant::now();
        let res = chunk_text(&MACOMBER.test_content, 1024, Some(10));
        let elapsed = start_time.elapsed();
        println!("{:?}", res);
        println!("{:?}", res.len());
        println!("{:?}", elapsed);
    }

    #[test]
    fn test_really_long() {
        let start_time = std::time::Instant::now();
        let res = chunk_text(&ROMEO_JULIET.test_content, 900, None);
        let elapsed = start_time.elapsed();
        println!("{:?}", res);
        println!("{:?}", res.len());
        println!("{:?}", elapsed);
    }
}
