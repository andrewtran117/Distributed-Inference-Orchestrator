from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    num_layers: int
    hidden_dim: int
    precision_bytes: int
    total_size_gb: float


PRESETS: dict[str, ModelConfig] = {
    "phi-3-3.8b-q4": ModelConfig(
        model_name="phi-3-3.8b-q4",
        num_layers=32, hidden_dim=3072, precision_bytes=2, total_size_gb=2.2,
    ),
    "mistral-7b-q4": ModelConfig(
        model_name="mistral-7b-q4",
        num_layers=32, hidden_dim=4096, precision_bytes=2, total_size_gb=3.9,
    ),
    "llama-3-8b-q4": ModelConfig(
        model_name="llama-3-8b-q4",
        num_layers=32, hidden_dim=4096, precision_bytes=2, total_size_gb=4.7,
    ),
    "qwen-2.5-14b-q4": ModelConfig(
        model_name="qwen-2.5-14b-q4",
        num_layers=48, hidden_dim=5120, precision_bytes=2, total_size_gb=8.2,
    ),
    "llama-3-70b-q4": ModelConfig(
        model_name="llama-3-70b-q4",
        num_layers=80, hidden_dim=8192, precision_bytes=2, total_size_gb=40.0,
    ),
    "qwen-2.5-72b-q4": ModelConfig(
        model_name="qwen-2.5-72b-q4",
        num_layers=80, hidden_dim=8192, precision_bytes=2, total_size_gb=41.0,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Look up a preset by name, or raise KeyError."""
    if name not in PRESETS:
        raise KeyError(f"Unknown model preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
