import json
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

model_list = [
    "01-ai/Yi-34B-Chat",
    "mistralai/Mistral-7B-v0.1",
    # "meta-llama/Llama-2-7b", #! Need waiting for access review
    # "meta-llama/Llama-2-7b-chat-hf", #! Need waiting for access review
]


def save_vocab(model_path: str, save_dir: Path) -> None:
    """Saves the vocabulary of a given model to a JSON file."""
    model_name = model_path.split("/")[-1].lower()
    save_path = save_dir / f"{model_name}_vocab.json"

    if save_path.exists():
        print(f"Vocab file for '{model_path}' already exists. Skipping...")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer for '{model_path}': {e}")
        return

    vocab = tokenizer.get_vocab()
    save_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=4))
    print(f"Vocab for '{model_path}' saved to {save_path}")


def main() -> None:
    save_dir = Path("vocab")
    save_dir.mkdir(exist_ok=True)

    for model_path in model_list:
        save_vocab(model_path, save_dir)


if __name__ == "__main__":
    main()
