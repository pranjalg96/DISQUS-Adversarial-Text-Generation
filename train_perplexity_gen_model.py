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

# Calc. perplexity, using validation set

parser = argparse.ArgumentParser(
    description='Train perplexity of Comment Generator'
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

p_word_dropout = 0.0
emb_dim = 100

dataset = DATA_DISQUS_SUP(batch_size=mb_size, emb_dim=emb_dim)

disqus_model = MODEL_DISQUS(
    dataset.n_vocab, h_dim, z_dim, num_layers_enc=num_layers_enc, bidirectional_enc=bidirectional_enc,
    dropout_enc=dropout_enc, p_word_dropout=p_word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=True, gpu=args.gpu
)

disqus_model.load_state_dict(torch.load('saved_models/disqus_model.bin'))
disqus_model.eval()

num_rows = 6558

num_batches = math.ceil(num_rows/mb_size)

dataset.create_batch_iterable()


def calc_perplexity(inputs):

    batch_size = inputs.size(1)
    # pad_words = Variable(torch.LongTensor([model.PAD_IDX])).repeat(1, batch_size)
    # pad_words = pad_words.cuda() if model.gpu else pad_words

    enc_inputs = inputs
    dec_inputs = inputs

    # Encoder: sentence -> z
    h_fc = disqus_model.forward_encoder(enc_inputs)
    z = disqus_model.sample_noise(batch_size)

    # Decoder: sentence -> y
    y_scores_raw = disqus_model.forward_generator(dec_inputs, h_fc, z)
    y_probs = F.softmax(y_scores_raw, dim=2)

    batch_perplexity = 0.0

    for sent_i in range(batch_size):

        sentence_perplexity = 0.0

        for word_j in range(disqus_model.MAX_SENT_LEN):

            prob_vector = y_probs[word_j, sent_i]
            input_token = inputs[word_j, sent_i]

            sentence_perplexity += -torch.log(prob_vector[int(input_token)]).item()

        sentence_perplexity /= disqus_model.MAX_SENT_LEN
        sentence_perplexity = np.exp(sentence_perplexity)
        batch_perplexity += sentence_perplexity

    batch_perplexity /= batch_size

    return batch_perplexity


def main():
    avg_perplexity = 0.0

    for batch_idx in range(num_batches):
        inputs, _ = dataset.next_batch(args.gpu)

        batch_perplexity = calc_perplexity(inputs)
        avg_perplexity += batch_perplexity

    avg_perplexity /= num_batches

    print('Average perplexity on Training set: ', np.exp(avg_perplexity))

    exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
