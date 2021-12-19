
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from FLICKR.sup_flickr8k_dataset import *
from FLICKR.model_flickr import RNN_VAE

import matplotlib.pyplot as plt

import argparse

# Measure perplexity on the validation set.

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()

mb_size = 32  # Mini-batch size
h_dim = 300
z_dim = 100
c_dim = 2

dataset = flickr8k_Data(emb_dim=200)

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
    gpu=args.gpu
)

num_val_rows = 550

num_val_batches = math.ceil(num_val_rows/mb_size)

# Load pretrained base VAE with c ~ p(c)
model.load_state_dict(torch.load('models/flickr8k_vae.bin'))
model.eval()

def calc_perplexity(model, inputs):

    batch_size = inputs.size(1)
    pad_words = Variable(torch.LongTensor([model.PAD_IDX])).repeat(1, batch_size)
    pad_words = pad_words.cuda() if model.gpu else pad_words

    enc_inputs = inputs
    dec_inputs = inputs

    # Encoder: sentence -> z
    mu, logvar = model.forward_encoder(enc_inputs)

    z = model.sample_z(mu, logvar)
    c = model.sample_c_prior(batch_size)

    # Decoder: sentence -> y
    y_scores_raw = model.forward_decoder(dec_inputs, z, c)
    y_probs = F.softmax(y_scores_raw, dim=2)

    batch_perplexity = 0.0

    for sent_i in range(batch_size):

        sentence_perplexity = 0.0

        for word_j in range(model.MAX_SENT_LEN):

            prob_vector = y_probs[word_j, sent_i]
            input_token = inputs[word_j, sent_i]

            sentence_perplexity += -torch.log(prob_vector[int(input_token)]).item()

        sentence_perplexity /= model.MAX_SENT_LEN
        batch_perplexity += sentence_perplexity

    batch_perplexity /= batch_size

    return batch_perplexity

def main():

    it = 0

    avg_perplexity = 0.0

    for batch_idx in range(num_val_batches):

        inputs, _ = dataset.next_validation_batch(args.gpu)

        batch_perplexity = calc_perplexity(model, inputs)
        avg_perplexity += batch_perplexity

    avg_perplexity /= num_val_batches

    print('Average perplexity on validation set: ', avg_perplexity)

if __name__ == '__main__':
    main()
