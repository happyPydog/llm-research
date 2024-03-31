import json
import click
from pathlib import Path
from transformers import AutoTokenizer


@click.command()
@click.argument("models", nargs=-1)
@click.option(
    "--target_directory",
    default="vocab",
    type=click.Path(),
    help='The directory where the vocabularies will be saved. Defaults to "vocab".',
)
def main(models: tuple[str, ...], target_directory: Path) -> None:
    """
    Downloads and saves the vocabularies for the specified models into the target directory.

    This function serves as the entry point for the command line interface. It iterates
    through the list of provided model names, constructs a save path for each, and then
    calls `save_vocab` to download and save the vocabulary file for each model.

    Args:
        models: A tuple of strings where each string is a model name to download vocabularies for.
        target_directory: A string representing the path to the directory where the vocabularies
            will be saved. If not specified, defaults to a directory named "vocab".
    """
    root_dir = Path(target_directory)

    for model_name in models:
        model_path = root_dir / Path(*model_name.split("/"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_vocab(model_name, model_path.with_suffix(".json"))


def save_vocab(model_name: str, save_path: Path) -> None:
    """
    Downloads the vocabulary for a given model and saves it to a specified path.

    This function attempts to download the vocabulary using the transformers library's
    AutoTokenizer. If successful, the vocabulary is saved in JSON format to the specified
    path. If the file already exists, the function will skip the download and notify the user.
    Any exceptions during the download or save process are caught and reported.

    Args:
        model_name: The name of the model to download the vocabulary for. This should be
            a model identifier recognized by the transformers library.
        save_path: A Path object representing the file path where the vocabulary JSON should
            be saved.
    """
    if save_path.exists():
        click.echo(f" Skip `{model_name}` because it already exists.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab = tokenizer.get_vocab()
        save_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=4))
        click.echo(f"Successfully saved vocab for '{model_name}' to {save_path}")
    except OSError as e:
        click.echo(f"Failed to save vocab for '{model_name}': {e}")


if __name__ == "__main__":
    main()
