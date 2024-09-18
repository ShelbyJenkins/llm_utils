use super::LocalLlmMetadata;

// Estimates from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/70c0241bc90e8025218a8d9667346aa72f60f472/LLMVRAMCalculator/LLMVRAMCalculator.py#L6

pub const Q8: f64 = 8.5;
pub const Q7: f64 = 7.5; // IDK
pub const Q6: f64 = 6.59;
pub const Q5: f64 = 5.90;
pub const Q4: f64 = 4.85;
pub const Q3: f64 = 3.91;
pub const Q2: f64 = 3.35;
pub const Q1: f64 = 2.0; // IDK

pub fn quantization_from_vram(base_model_bytes: f64, vram_gb: u32, context_overhead: f64) -> u8 {
    let cuda_overhead_gb = 0.000; // I don't think this is required.

    let memory_gb = vram_gb as f64 - cuda_overhead_gb - context_overhead;
    let memory_bytes = memory_gb * 1024.0 * 1024.0 * 1024.0;

    // let model_bytes = num_params * bpw / 8
    let estimate_quantized_model_size =
        |base_model_bytes: f64, q_bits: f64| base_model_bytes * q_bits / 8.0;

    match memory_bytes {
        x if x >= estimate_quantized_model_size(base_model_bytes, Q8) => 8,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q7) => 7,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q6) => 6,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q5) => 5,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q4) => 4,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q3) => 3,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q2) => 2,
        x if x >= estimate_quantized_model_size(base_model_bytes, Q1) => 1,
        _ => panic!("Not enough VRAM!"),
    }
}

// This is converted from https://github.com/pandora-s-git/LLMVRAMCalculator/blob/main/LLMVRAMCalculator/LLMVRAMCalculator.py
pub fn estimate_context_size(
    model_metadata: &LocalLlmMetadata,
    ctx_size: u64,
    batch_size: u64,
) -> f64 {
    let input_buffer = input_buffer(model_metadata, ctx_size, batch_size);
    let compute_buffer = compute_buffer(model_metadata, ctx_size);
    let kv_cache = kv_cache(model_metadata, ctx_size);
    let context_bits = input_buffer + kv_cache + compute_buffer;
    context_bits / (1024f64 * 1024f64 * 1024f64)
}

fn input_buffer(model_metadata: &LocalLlmMetadata, ctx_size: u64, batch_size: u64) -> f64 {
    ((batch_size * 3)
        + (model_metadata.hidden_size * batch_size)
        + (batch_size * ctx_size)
        + ctx_size) as f64
}

fn compute_buffer(model_metadata: &LocalLlmMetadata, ctx_size: u64) -> f64 {
    (ctx_size as f64 / 1024f64 * 2f64 + 0.75)
        * model_metadata.num_attention_heads as f64
        * 1024f64
        * 1024f64
}

fn kv_cache(model_metadata: &LocalLlmMetadata, ctx_size: u64) -> f64 {
    let cache_bit = match model_metadata.torch_dtype.as_str() {
        "float32" => 32,
        "float16" | "bfloat16" => 16,
        _ => panic!("Unsupported data type"),
    };
    let n_gqa = model_metadata.num_attention_heads / model_metadata.num_key_value_heads;
    let n_embd_gqa = model_metadata.hidden_size / n_gqa;
    let n_elements = n_embd_gqa * (model_metadata.num_hidden_layers * ctx_size);
    let size = 2 * n_elements;
    size as f64 * (cache_bit as f64 / 8f64)
}
