use lazy_static::lazy_static;
use std::{fs, path::PathBuf};

macro_rules! generate_structs {
    ($($file:ident),+) => {
        $(
            #[allow(non_camel_case_types)]
            pub struct $file {
                pub test_content: String,
                pub sentence_splits: Vec<String>,
            }
            impl $file {
                pub fn load() -> Self {
                    let file_name = format!("{}.toml", stringify!($file).to_lowercase());
                    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                    let file_path=  PathBuf::from(cargo_manifest_dir)
                        .join("src")
                        .join("text_utils")
                        .join("test_text")
                        .join("files")
                        .join(file_name);

                    let content = fs::read_to_string(&file_path).expect("Failed to read file");
                    let data: toml::Value = toml::from_str(&content).expect("Failed to parse TOML");

                    Self {
                        test_content: data["test_content"].as_str().unwrap().to_string(),
                        sentence_splits: data.get("sentence_splits")
                            .and_then(|value| value.as_array())
                            .map(|array| {
                                array.iter()
                                    .map(|s| s.as_str().unwrap().to_string())
                                    .collect()
                            })
                            .unwrap_or_default(),
                    }
                }
            }
        )+
    };
}

generate_structs!(Bnf);
generate_structs!(Doom);
generate_structs!(Long);
generate_structs!(Romeo_Juliet);
generate_structs!(Macomber);

generate_structs!(Turing);
generate_structs!(Shake);
generate_structs!(Katana);

lazy_static! {
    pub static ref BNF: Bnf = Bnf::load();
    pub static ref DOOM: Doom = Doom::load();
    pub static ref LONG: Long = Long::load();
    pub static ref MACOMBER: Macomber = Macomber::load();
    pub static ref ROMEO_JULIET: Romeo_Juliet = Romeo_Juliet::load();
    pub static ref TURING: Turing = Turing::load();
    pub static ref SHAKE: Shake = Shake::load();
    pub static ref KATANA: Katana = Katana::load();
}
