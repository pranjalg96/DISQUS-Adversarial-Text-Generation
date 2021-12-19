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


def main():

    for sent_i in range(num_valid_examples):

        x, true_label = valid_inputs[:, sent_i].unsqueeze(dim=1), valid_labels[sent_i]
        h_fc = disqus_model.forward_encoder(x)

        sample_idxs = disqus_model.sample_comment(h_fc, raw=True)
        # sample_idxs = Variable(torch.LongTensor(sample_idxs)).unsqueeze(0)
        # sample_idxs = sample_idxs.cuda() if args.gpu else sample_idxs

        y_score_raw = disqus_model.forward_classifier(sample_idxs)
        y_score = F.softmax(y_score_raw, dim=1)
        _, y_pred = torch.max(y_score, dim=1)
        y_pred_label = dataset.idx2label(y_pred)

        sample_idxs = sample_idxs.squeeze(0)

        sample_sent = dataset.idxs2sentence_suppressed(disqus_model, sample_idxs)
        orig_sent = dataset.idxs2sentence_suppressed(disqus_model, x.squeeze(dim=1))

        print('Original sentence: "{}", "{}"'.format(orig_sent, dataset.idx2label(true_label)))
        print('Sampled sentence: "{}", "{}"'.format(sample_sent, y_pred_label))
        print()

    exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
