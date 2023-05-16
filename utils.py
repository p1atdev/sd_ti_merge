import torch
from safetensors.torch import load_file, save_file
from pathlib import Path


def load_model(model_path: Path):
    suffix = model_path.suffix
    if suffix == ".safetensors":
        model = load_file(model_path, device="cpu")
    else:
        model = torch.load(model_path, map_location="cpu")
    return model


def save_model(model, output_path: Path):
    suffix = output_path.suffix
    if suffix == ".safetensors":
        save_file(model, output_path)
    else:
        torch.save(model, output_path)


def get_emb_params(model):
    try:
        return model["string_to_param"]["*"]  # AUTOMATIC1111 style
    except:
        pass
    try:
        return model["emb_params"]  # kohya-ss style
    except:
        pass

    raise ValueError(f"Invalid model format. / モデルの形式が不正です。")
