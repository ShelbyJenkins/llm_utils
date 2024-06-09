use anyhow::Result;
use article_scraper::Readability;
use html2text::{config, render::text_renderer::TrivialDecorator};

/// This seems to have to be async, even though it doesn't need to be
pub async fn clean_html(html: &str) -> Result<String> {
    // Render to html using article_scraper's implementation of Mozilla's Readability mode
    let readable = Readability::extract(html, None).await?;
    // Convert html to text with html2text
    // Trivial decorator removes all tags and leaves only text
    let decorator = TrivialDecorator::new();
    let text = config::with_decorator(decorator)
        .allow_width_overflow()
        .string_from_read(readable.as_bytes(), 10000)
        .unwrap();
    // Finally, remove unwanted chars and excess whitespace
    Ok(super::clean_text::TextCleaner::new().run(&text))
}
