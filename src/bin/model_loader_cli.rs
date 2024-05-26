use anyhow::Result;
use llm_utils::models::gguf::GGUFModelBuilder;

// cargo run -p llm_utils --bin model_loader_cli -- --model_url "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"

#[tokio::main]
pub async fn main() -> Result<()> {
    let matches = clap::Command::new("Model Downloader")
        .version("1.0")
        .about("Downloads and sets up models")
        .arg(
            clap::Arg::new("model_url")
                .help("The model URL")
                .long("model_url")
                .required(true),
        )
        .arg(
            clap::Arg::new("hf_token")
                .help("HF token")
                .long("hf_token")
                .required(false),
        )
        .get_matches();

    let model_url = matches.get_one::<String>("model_url").unwrap();
    let hf_token = matches
        .get_one::<String>("hf_token")
        .map(|token| token.to_owned());
    GGUFModelBuilder::new(hf_token)
        .from_quant_file_url(model_url.as_str())
        .load()
        .await?;

    Ok(())
}
