from dataclasses import dataclass


@dataclass(frozen=True)
class MLAConfig:
    # Multi-head latent attention settings.
    q_lora_rank: int = 384
    kv_lora_rank: int = 128
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64


@dataclass(frozen=True)
class ModelConfig:
    # RTX 5090-friendly toy base model: big enough to learn coherent text from
    # Wikipedia, small enough to pretrain for fun on one 32 GB consumer GPU.
    name: str = "wiki-texting-220m-mla-rtx5090"
    target_params: int = 220_000_000

    # Tokenizer / sequence.
    tokenizer_type: str = "bpe"
    tokenizer_train_corpus: str = "wikipedia"
    tokenizer_model_path: str = "data/tokenizer/wiki_bpe_32k.model"
    vocab_size: int = 32_000
    max_seq_len: int = 1024

    # Llama-style transformer body, with MLA replacing standard attention.
    n_layers: int = 18
    n_heads: int = 12
    n_kv_heads: int = 4
    hidden_size: int = 768
    intermediate_size: int = 2048
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    mla: MLAConfig = MLAConfig()

    # Training defaults for a single RTX 5090 baseline.
    device: str = "cuda"
    gpu_name: str = "rtx_5090"
    gpu_vram_gb: int = 32
    compile: bool = True
    activation_checkpointing: bool = True
    flash_attention: bool = True
    batch_size: int = 16
    gradient_accumulation_steps: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 6_000
    dtype: str = "bfloat16"
    total_pretrain_tokens: int = 3_000_000_000

    # Data curriculum.
    pretrain_dataset: str = "wikipedia"
    pretrain_dataset_source: str = "huggingface:wikimedia/wikipedia"
    pretrain_dataset_config: str = "20231101.en"
    raw_wikipedia_dump_url: str = (
        "https://dumps.wikimedia.org/enwiki/latest/"
        "enwiki-latest-pages-articles-multistream.xml.bz2"
    )
    pretrain_objective: str = "base_model_for_fun"
    finetune_dataset: str = "personal_text_messages"
    finetune_objective: str = "texting_style_imitation"
    finetune_learning_rate: float = 5e-5
    finetune_steps: int = 1_000
    lora_finetune: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64


config = ModelConfig()


if __name__ == "__main__":
    print(config)
