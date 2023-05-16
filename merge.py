import torch
import numpy as np
import argparse
from pathlib import Path
from utils import load_model, save_model, get_emb_params


def main(args):
    model_paths = args.model_paths
    output_path = Path(args.output_path).resolve()
    ratios = args.ratios

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path.name} already exists. Use '--overwrite' or '-w' to overwite. / {output_path.name} は既に存在します。'--overwrite' か '-w' を指定すると上書きします。"
        )

    if len(model_paths) == 1:
        raise ValueError(f"Number of models must be more than 1. / モデル数は2以上である必要があります。")

    if ratios is None:
        ratios = [1] * len(model_paths)

    if len(model_paths) != len(ratios):
        raise ValueError(
            f"Number of models and ratios must be same. / モデル数と割合の数は同じである必要があります。"
        )

    model_paths = [Path(model_path).resolve() for model_path in model_paths]

    shape = None

    models = []
    for model_path in model_paths:
        model = get_emb_params(load_model(model_path))
        print(f"{model_path.name}: {model.shape[0]} tokens")
        if shape is None:
            shape = model.shape
        else:
            if shape != model.shape:
                raise ValueError(
                    f"Mismatched token length. (expected: {shape[0]}, {model_path.name}: {model.shape[0]}) / トークン長が一致しません。(期待値: {shape[0]}, {model_path.name}: {model.shape[0]})"
                )

        models.append(model)

    print(f"Merging... / モデルをマージしています...")

    merged_model = torch.zeros(shape)

    for model, ratio in zip(models, ratios):
        merged_model += model * ratio

    merged_model /= np.sum(ratios)

    print(f"Saving... / 保存しています...")

    save_model(
        {"emb_params": merged_model},
        output_path,
    )

    print("Done! / 完了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_paths",
        type=str,
        nargs="+",
        help="models to merge",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="output path",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help="ratios to merge models",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="overwrite output file",
    )
    args = parser.parse_args()

    main(args)
