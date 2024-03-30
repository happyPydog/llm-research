import json
from pathlib import Path
from transformers import AutoTokenizer

# Ensure ROOT_DIR exists at the start
ROOT_DIR = Path("vocab")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LIST = [
    "01-ai/Yi-34B-Chat",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "lmsys/vicuna-13b-v1.5",
    "Nexusflow/Starling-LM-7B-beta",
]


def generate_save_path(model_path: str, root_dir: Path) -> Path:
    """Generates the save path for a model's vocabulary based on the model path."""
    _, model_name = model_path.split("/")
    return root_dir / f"{model_name}.json"


def save_vocab(model_path: str, save_path: Path) -> None:
    """Saves the vocabulary of a given model if it doesn't already exist."""
    if save_path.exists():
        print(f"Vocab file {save_path} already exists. Skipping...")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = tokenizer.get_vocab()
        save_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=4))
        print(f"Successfully saved vocab for '{model_path}' to {save_path}")
    except Exception as e:
        print(f"Failed to save vocab for '{model_path}': {e}")


def main() -> None:
    for model_path in MODEL_LIST:
        save_path = generate_save_path(model_path, ROOT_DIR)
        save_vocab(model_path, save_path)


if __name__ == "__main__":
    main()
