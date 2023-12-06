from typing import Tuple
import numpy as np
import tqdm as tqdm
import utils
from transformer import Seq2Seq
from train import get_model
import re
import torch
import os 
import pickle

MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"

def load_model(dirpath, model_params, hyperparameters, device, model_path):
    # Global Configuration and Parameters
    with open(os.path.join(dirpath, 'vocab', "src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    with open(os.path.join(dirpath, 'vocab', "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)
    model = get_model(src_lang, trg_lang, model_params, device, hyperparameters) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def predict(model,function: str):
    v_ptr = r"sin|cos|tan|exp|\^|\d+|\w|\(|\)|\+|-|\*+|\/"
    token=re.findall(v_ptr, function)
    prediction_tensors = []
    prediction_tensors.append(utils.sentence_to_tensor(token, model.src_lang))
    preds, _, _ = model.predict(prediction_tensors)
    return preds[0]
# ----------------- END ----------------- #
def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    model_params = {
        'max_len': MAX_SEQUENCE_LENGTH,
        'hid_dim': 64,
        'enc_layers': 2,
        'dec_layers': 2,
        'enc_heads': 4,
        'dec_heads': 4,
        'enc_pf_dim': 128,
        'dec_pf_dim': 128,
        'enc_dropout': 0.3,
        'dec_dropout': 0.3,
    }
    hyperparameters = {
        'batch_size': 1,
        'n_iters': 20,
        'gradient_clip': 1,
        'learning_rate': 0.0001
    }
    device = utils.get_device()
    dirpath = os.getcwd()
    model_path = os.path.join(os.getcwd(), 'model', 'best_model.pt')
    model = load_model(dirpath, model_params, hyperparameters, device, model_path)
    functions, true_derivatives = load_file(filepath)
    predicted_derivatives = [predict(model, f) for f in functions]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
