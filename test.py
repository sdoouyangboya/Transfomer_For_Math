#!/usr/bin/env python
# coding=utf-8

"""
Evaluating the model on test.txt
1) Load the trained model from ./model
2) Calculate and report the loss on test data
3) Generate batch predictions for all test data (Much faster than predicting one by one)
4) Store the predictions in an output file for documentation
5) Do a strict string equality match between predictions and actual sequences
6) Report the accuracy using the formula: accuracy = (correct predictions / total test samples)*100
"""

# Imports
import os
import pickle
import tqdm as tqdm
import logging
import utils
from transformer import Seq2Seq
from train import get_model
from data import load_pairs
import torch

logger = logging.getLogger("TestLogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")



def test_pred_accuracy(model, test_data):
    """
    Generates predictions, compares with the ground truth values and reports the accuracy
    """
    prediction_tensors = []
    labels = []
    correct_count = 0
    total_count = len(test_data)

    for pair in test_data:
        prediction_tensors.append(utils.sentence_to_tensor(pair[0], model.src_lang))
        labels.append("".join(pair[1]))
    
    preds, _, _ = model.predict(prediction_tensors)
    for pred, actual in zip(preds, labels):
        if pred == actual:
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy, preds


def load_model(dirpath, model_params, hyperparameters, device, model_path):
    # Global Configuration and Parameters

    logger.info("Loading source vocabulary")
    with open(os.path.join(dirpath, 'vocab', "src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    logger.info("Finished Loading source vocabulary!")
    logger.info("Loading Target vocabulary")
    with open(os.path.join(dirpath, 'vocab', "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)
    logger.info("Finished Loading Target vocabulary!")
    model = get_model(src_lang, trg_lang, model_params, device, hyperparameters) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def run_test(data_path):
    data_path = os.getcwd()
    logger.info("Loading the trained model!")
    device = utils.get_device()
    model_path = os.path.join(os.getcwd(), 'model', 'best_model.pt')
    dirpath = os.getcwd()
    model_params = {
        'max_len': 30,
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
        'batch_size': 512,
        'n_iters': 20,
        'gradient_clip': 1,
        'learning_rate': 0.0001
    }
    model = load_model(dirpath, model_params, hyperparameters, device, model_path)
    logger.info("Finished Loading the trained model!")
    
    logger.info("Loading the test data!")
    test_pairs = load_pairs(data_path, test_flag=True)
    logger.info("Finished loading the test data!")
    
    logger.info("Started batch predictions on test data!")
    accuracy, predictions = test_pred_accuracy(model, test_pairs)
    logger.info("Finished batch predictions on test data!")
    
    print("The model accuracy on test data = {}%".format(accuracy * 100))
    if not os.path.exists(os.path.join(os.getcwd(), 'output')):
        os.mkdir(os.path.join(data_path, 'output'))
    
    with open(os.path.join(data_path, 'output', 'predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    logger.info("Finished saving predictions!")


if __name__ == '__main__':
    data_path = os.getcwd()  # or specify the appropriate path
    run_test(data_path)
