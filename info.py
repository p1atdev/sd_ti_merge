from utils import load_model
from pathlib import Path
import argparse


def main(args):
    model_path = Path(args.model_path)

    model = load_model(model_path)

    print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="model path",
    )
    args = parser.parse_args()

    main(args)
