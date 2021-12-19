
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from DISQUS.dataset_DISQUS_LM import DATA_DISQUS_LM
from DISQUS.model_DISQUS import MODEL_DISQUS

import matplotlib.pyplot as plt

import argparse

# Completely teacher-forcing during training the generator and student-forcing during sampling. Can try to change that.

parser = argparse.ArgumentParser(
    description='Conditional Comment Generator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()

mb_size = 32  # Mini-batch size

h_dim = 300
z_dim = 100

lr = 1e-3
lr_decay_every = 10000  # Consider using smaller number here

n_epochs = 100
log_interval = 100

num_layers_enc = 2
bidirectional_enc = True
dropout_enc = 0.3

p_word_dropout = 0.3
emb_dim = 100

dataset = DATA_DISQUS_LM(batch_size=mb_size, emb_dim=emb_dim)

model = MODEL_DISQUS(
    dataset.n_vocab, h_dim, z_dim, num_layers_enc=num_layers_enc, bidirectional_enc=bidirectional_enc,
    dropout_enc=dropout_enc, p_word_dropout=p_word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=False, gpu=args.gpu
)

num_rows = 6558

num_batches = math.ceil(num_rows/mb_size)

dataset.create_batch_iterable()
recon_losses = []


def main():

    trainer = optim.Adam(model.model_params, lr=lr)

    it = 0

    for epoch in range(n_epochs):

        avg_recon_loss = 0.0

        for batch_idx in range(num_batches):
            trainer.zero_grad()

            inputs = dataset.next_batch(args.gpu)

            recon_loss = model.forward(inputs)
            avg_recon_loss += recon_loss.item()

            recon_loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)  # check this

            cur_batch_size = inputs.size(1)

            trainer.step()

            if it % log_interval == 0:
                rand_idx = np.random.randint(0, cur_batch_size)

                x = inputs[:, rand_idx].unsqueeze(dim=1)  # select random sentence from minibatch for sampling
                h_fc = model.forward_encoder(x)

                sample_idxs = model.sample_comment(h_fc)

                sample_sent = dataset.idxs2sentence(sample_idxs)
                orig_sent = dataset.idxs2sentence_suppressed(model, x.squeeze(dim=1))

                print('Iter-{}; Recon: {:.4f}'
                      .format(it, recon_loss.item()))
                print('Original sentence: "{}"'.format(orig_sent))
                print('Sampled sentence: "{}"'.format(sample_sent))
                print()

            # Anneal learning rate
            new_lr = lr * (0.5 ** (it // lr_decay_every))
            for param_group in trainer.param_groups:
                param_group['lr'] = new_lr

            it += 1

        avg_recon_loss /= num_batches
        recon_losses.append(avg_recon_loss)

    plt.plot(recon_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Recon loss')
    plt.show()

    if args.save:
        save_model()

    exit(0)


def save_model():
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    torch.save(model.state_dict(), 'saved_models/cond_comment_gen.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:

        plt.plot(recon_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Recon loss')
        plt.show()

        if args.save:
            save_model()

        exit(0)
