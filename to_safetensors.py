import torch
from safetensors.torch import save_file
import argparse
from pathlib import Path


def main(args):
    input_path = Path(args.input_path).resolve()
    output_path = args.output_path
    overwrite = args.overwrite

    if input_path.suffix == ".safetensors":
        raise ValueError(
            f"{input_path} is already a safetensors file. / {input_path} は既に safetensors ファイルです。"
        )

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.safetensors"
    else:
        output_path = Path(output_path).resolve()

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path.name} already exists. Use '--overwrite' or '-w' to overwite. / {output_path.name} は既に存在します。'--overwrite' か '-w' を指定すると上書きします。"
        )

    print(f"Loading...")

    model = torch.load(input_path, map_location="cpu")
    save_file(model, output_path)

    print("Done!")

    print(f"Saved to {output_path} /\n {output_path} に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="input path",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="overwrite output file",
    )
    args = parser.parse_args()

    main(args)
