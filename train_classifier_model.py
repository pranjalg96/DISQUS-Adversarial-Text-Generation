
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

# Using same architecture as discriminator (CNN)

parser = argparse.ArgumentParser(
    description='Toxic comment classifier'
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

n_epochs = 15
log_interval = 100

num_layers_enc = 2
bidirectional_enc = True
dropout_enc = 0.3

p_word_dropout = 0.3
emb_dim = 100

dataset = DATA_DISQUS_SUP(batch_size=mb_size, emb_dim=emb_dim)

model = MODEL_DISQUS(
    dataset.n_vocab, h_dim, z_dim, num_layers_enc=num_layers_enc, bidirectional_enc=bidirectional_enc,
    dropout_enc=dropout_enc, p_word_dropout=p_word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=True, gpu=args.gpu
)

model.load_state_dict(torch.load('saved_models/cond_comment_gen.bin'))
model.train()

num_rows = 6558

num_batches = math.ceil(num_rows/mb_size)

dataset.create_batch_iterable()

classifier_losses = []
classifier_accuracies = []


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

    trainer = optim.Adam(model.classifier_params, lr=lr)

    avg_classifier_accuracy = None
    avg_classifier_precision = None
    avg_classifier_recall = None

    it = 0

    for epoch in range(n_epochs):

        avg_classifier_loss = 0.0
        avg_classifier_accuracy = 0.0

        avg_classifier_precision = 0.0
        avg_classifier_recall = 0.0

        for batch_idx in range(num_batches):
            # trainer.zero_grad()

            inputs, labels = dataset.next_batch(args.gpu)

            y_scores_raw = model.forward_classifier(inputs.transpose(0, 1))
            classifier_loss = F.cross_entropy(y_scores_raw, labels)

            avg_classifier_loss += classifier_loss.item()

            classifier_acc, classifier_precision, classifier_recall = calc_metrics(y_scores_raw, labels)

            avg_classifier_accuracy += classifier_acc
            avg_classifier_precision += classifier_precision
            avg_classifier_recall += classifier_recall

            classifier_loss.backward()
            trainer.step()

            trainer.zero_grad()

            if it % log_interval == 0:

                print('Iter-{}; Loss: {:.4f}, Acc: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'
                      .format(it, classifier_loss.item(), classifier_acc, classifier_precision, classifier_recall))
                print()

            # Anneal learning rate
            new_lr = lr * (0.5 ** (it // lr_decay_every))
            for param_group in trainer.param_groups:
                param_group['lr'] = new_lr

            it += 1

        avg_classifier_loss /= num_batches
        avg_classifier_accuracy /= num_batches
        avg_classifier_precision /= num_batches
        avg_classifier_recall /= num_batches

        classifier_losses.append(avg_classifier_loss)
        classifier_accuracies.append(avg_classifier_accuracy)

    plt.plot(classifier_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Classifier loss')
    plt.show()

    plt.plot(classifier_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Classifier Accuracy')
    plt.show()

    print('Final classifier accuracy: ', avg_classifier_accuracy)
    print('Final classifier precision: ', avg_classifier_precision)
    print('Final classifier recall: ', avg_classifier_recall)

    if args.save:
        save_model()

    exit(0)


def save_model():
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    torch.save(model.state_dict(), 'saved_models/toxic_classifier_model.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:

        plt.plot(classifier_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Classifier loss')
        plt.show()

        plt.plot(classifier_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Classifier Accuracy')
        plt.show()

        if args.save:
            save_model()

        exit(0)
