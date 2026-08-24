from glob import glob
from pathlib import Path

HF_CACHE = Path(__file__).resolve().parent.parent / "hf-cache"


def default_model_files():
    classifier = glob(str(
        HF_CACHE / "models--typeform--distilbert-base-uncased-mnli/snapshots/*/model.safetensors"
    ))
    snapshots = glob(str(
        HF_CACHE / "models--Efficient-Large-Model--VILA1.5-3b-AWQ/snapshots/*"
    ))
    vision = [s + "/vision_tower/model.safetensors" for s in snapshots]
    llm = [p for s in snapshots for p in glob(s + "/llm/*.pt")]
    files = classifier[:1] + vision[:1] + llm[:1]
    if len(files) != 3:
        raise FileNotFoundError(f"expected 3 model files under {HF_CACHE}, found {files}")
    return files
