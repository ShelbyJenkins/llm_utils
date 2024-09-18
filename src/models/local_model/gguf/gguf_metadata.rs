use super::gguf_file;
use anyhow::Context;
use std::{collections::HashMap, path::PathBuf};

pub struct GgufMetadata {
    pub contents: Vec<gguf_file::Content>,
    pub architecture: String,
    metadata: HashMap<String, gguf_file::Value>,
}

impl GgufMetadata {
    pub fn from_path(path: &PathBuf) -> crate::Result<Self> {
        let mut reader = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let metadata = content.metadata.clone();

        let architecture = get_value(&["general"], "architecture", &metadata)?;

        Ok(Self {
            contents: vec![content],
            architecture,
            metadata,
        })
    }

    pub fn from_paths(paths: &[PathBuf]) -> crate::Result<Self> {
        let mut readers = Vec::new();
        for filename in paths {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers: Vec<&mut std::fs::File> = readers.iter_mut().collect::<Vec<_>>();

        let mut contents = Vec::new();
        let n_readers = readers.len();
        for reader in readers.iter_mut() {
            contents.push(gguf_file::Content::read(reader)?);
        }
        let n_splits = contents
            .iter()
            .filter_map(|ct| {
                ct.metadata
                    .get("split.count")
                    .map(|val| val.to_u64().unwrap())
            })
            .fold(Vec::new(), |mut accum, x| {
                if !accum.contains(&x) {
                    accum.push(x);
                }
                accum
            });
        if n_splits.len() > 1 {
            crate::bail!("GGUF files have differing `split.count` values: {n_splits:?}. Perhaps the GGUF files do not match?");
        }
        #[allow(clippy::cast_possible_truncation)]
        if !n_splits.is_empty() && n_readers != n_splits[0] as usize {
            crate::bail!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            );
        } else if n_splits.len() == 1 {
            println!("GGUF file has been split into {} shards", n_splits[0]);
        }
        let mut architecture = None;
        for ct in &contents {
            if !ct.metadata.contains_key("general.architecture") {
                continue;
            }

            architecture = Some(
                ct.metadata["general.architecture"]
                    .to_string()
                    .context("Model metadata should have declared an architecture")?
                    .clone(),
            );
        }

        let architecture = architecture.expect("GGUF files must specify `general.architecture`");
        let mut all_metadata = HashMap::new();
        for content in &contents {
            all_metadata.extend(content.metadata.clone())
        }

        Ok(Self {
            contents,
            architecture,
            metadata: all_metadata,
        })
    }

    pub fn get_value<T>(&self, path_prefixes: &[&str], field_name: &str) -> Result<T, crate::Error>
    where
        T: TryFrom<gguf_file::Value, Error = crate::Error>,
    {
        get_value(path_prefixes, field_name, &self.metadata)
    }

    pub fn get_option_value<T>(
        &self,
        path_prefixes: &[&str],
        field_name: &str,
    ) -> Result<Option<T>, crate::Error>
    where
        T: TryFrom<gguf_file::Value, Error = crate::Error>,
    {
        get_option_value(path_prefixes, field_name, &self.metadata)
    }
}

fn get_value<T>(
    path_prefixes: &[&str],
    field_name: &str,
    metadata: &HashMap<String, gguf_file::Value>,
) -> Result<T, crate::Error>
where
    T: TryFrom<gguf_file::Value, Error = crate::Error>,
{
    let prop_key = if path_prefixes.is_empty() {
        field_name.to_string()
    } else {
        let prefix = path_prefixes.join(".");
        format!("{}.{}", prefix, field_name)
    };
    metadata
        .get(&prop_key)
        .ok_or_else(|| anyhow::anyhow!("Key not found: `{prop_key}`",))
        .and_then(|value| T::try_from(value.clone()))
}

fn get_option_value<T>(
    path_prefixes: &[&str],
    field_name: &str,
    metadata: &HashMap<String, gguf_file::Value>,
) -> Result<Option<T>, crate::Error>
where
    T: TryFrom<gguf_file::Value, Error = crate::Error>,
{
    let prop_key = if path_prefixes.is_empty() {
        field_name.to_string()
    } else {
        let prefix = path_prefixes.join(".");
        format!("{}.{}", prefix, field_name)
    };

    match metadata.get(&prop_key) {
        Some(value) => T::try_from(value.clone()).map(Some),
        None => Ok(None),
    }
}

macro_rules! impl_try_from {
    ($($t:ty, $method:ident);*) => {
        $(
            impl TryFrom<gguf_file::Value> for $t {
                type Error = crate::Error;
                fn try_from(value: gguf_file::Value) -> Result<Self, Self::Error> {
                    value.$method().map_err(crate::Error::from)
                }
            }
        )*
    };
}

impl_try_from!(
    u8, to_u8;
    i8, to_i8;
    u16, to_u16;
    i16, to_i16;
    u32, to_u32;
    i32, to_i32;
    i64, to_i64;
    f32, to_f32;
    f64, to_f64;
    bool, to_bool
);

impl TryFrom<gguf_file::Value> for u64 {
    type Error = crate::Error;

    fn try_from(value: gguf_file::Value) -> Result<Self, Self::Error> {
        value.to_u64().map_err(crate::Error::from)
    }
}

impl TryFrom<gguf_file::Value> for String {
    type Error = crate::Error;
    fn try_from(value: gguf_file::Value) -> Result<Self, Self::Error> {
        value
            .to_string()
            .map(|s| s.to_owned())
            .map_err(crate::Error::from)
    }
}

impl<T> TryFrom<gguf_file::Value> for Vec<T>
where
    T: TryFrom<gguf_file::Value, Error = crate::Error>,
{
    type Error = crate::Error;

    fn try_from(value: gguf_file::Value) -> Result<Self, Self::Error> {
        match value {
            gguf_file::Value::Array(vec) => vec.into_iter().map(T::try_from).collect(),
            _ => Err(anyhow::anyhow!("Expected Array, found {:?}", value)),
        }
    }
}
