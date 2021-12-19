import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from DISQUS.dataset_DISQUS_sup import DATA_DISQUS_SUP
from DISQUS.model_DISQUS import MODEL_DISQUS

import matplotlib.pyplot as plt

import argparse

# Generate sentences from the complete model, using validation set

parser = argparse.ArgumentParser(
    description='Validation of Comment Generator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()

mb_size = 32  # Mini-batch size

h_dim = 300
z_dim = 100

num_layers_enc = 2
bidirectional_enc = True
dropout_enc = 0.3

p_word_dropout = 0.3
emb_dim = 100

dataset = DATA_DISQUS_SUP(batch_size=mb_size, emb_dim=emb_dim)

disqus_model = MODEL_DISQUS(
    dataset.n_vocab, h_dim, z_dim, num_layers_enc=num_layers_enc, bidirectional_enc=bidirectional_enc,
    dropout_enc=dropout_enc, p_word_dropout=p_word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=True, gpu=args.gpu
)

disqus_model.load_state_dict(torch.load('saved_models/disqus_model.bin'))
disqus_model.eval()

valid_inputs, valid_labels = dataset.next_validation_batch()

num_valid_examples = valid_inputs.size(1)
print('Number of validation examples: ', num_valid_examples)

def calc_metrics(y_scores_raw, y_true):

    y_scores = F.softmax(y_scores_raw, dim=1)
    _, y_preds = torch.max(y_scores, dim=1)

    y_preds_list = y_preds.tolist()
    y_true_list = y_true.data.tolist()

    y_preds_labels = [dataset.idx2label(i) for i in y_preds_list]
    y_true_labels = [dataset.idx2label(i) for i in y_true_list]

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for y_pred_i, y_true_i in zip(y_preds_labels, y_true_labels):

        if y_pred_i == 'toxic' and y_true_i == 'toxic':
            TP += 1
        elif y_pred_i == 'toxic' and y_true_i == 'non_toxic':
            FP += 1
        elif y_pred_i =='non_toxic' and y_true_i == 'toxic':
            FN += 1
        else:
            TN += 1

    accuracy = (TP + TN)/(TP + FP + TN + FN)
    precision = 0
    recall = 0

    if TP + FP != 0:
        precision = TP/(TP + FP)

    if TP + FN != 0:
        recall = TP/(TP + FN)

    return accuracy, precision, recall


def main():

    y_scores_raw = disqus_model.forward_classifier(valid_inputs.transpose(0, 1))

    classifier_acc, classifier_precision, classifier_recall = calc_metrics(y_scores_raw, valid_labels)

    print('Classifier validation accuracy: ', classifier_acc)
    print('Classifier validation precision: ', classifier_precision)
    print('Classifier validation recall: ', classifier_recall)

    exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
