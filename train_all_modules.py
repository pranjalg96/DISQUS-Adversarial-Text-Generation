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

# Need to implement MMD part still (could consider implementing a sentence similarity loss function instead)
# Instead of sampling from the multinomial distribution at each step during the log_interval, could try to implement
# beam search to get high probability sentences. (lower temperature is more of a greedy search approach)
# Can also consider adding teacher-forcing MLE loss again

parser = argparse.ArgumentParser(
    description='Training of all the modules of the DISQUS adversarial comment generator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()

mb_size = 32  # Mini-batch size

h_dim = 300
z_dim = 100

lr = 1  # 1e-3
lr_decay_every = 10000

n_epochs = 50
log_interval = 100

num_layers_enc = 2
bidirectional_enc = True
dropout_enc = 0.3

p_word_dropout = 0.3
emb_dim = 100

temp = 1  # Initial temperature
temp_decay_every = 2000

dataset = DATA_DISQUS_SUP(batch_size=mb_size, emb_dim=emb_dim)

disqus_model = MODEL_DISQUS(
    dataset.n_vocab, h_dim, z_dim, num_layers_enc=num_layers_enc, bidirectional_enc=bidirectional_enc,
    dropout_enc=dropout_enc, p_word_dropout=p_word_dropout, pretrained_embeddings=dataset.get_vocab_vectors(),
    freeze_embeddings=True, gpu=args.gpu
)

disqus_model.load_state_dict(torch.load('saved_models/toxic_classifier_model.bin'))
disqus_model.train()

num_rows = 6558
num_batches = math.ceil(num_rows/mb_size)

dataset.create_batch_iterable()

gen_losses = []
disc_losses = []

adv_gen_losses = []
adv_classifier_accuracies = []


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

    trainer_G = optim.Adam(disqus_model.gen_params, lr=lr)
    trainer_D = optim.Adam(disqus_model.discriminator_params, lr=lr)

    trainer_G.zero_grad()
    trainer_D.zero_grad()

    new_temp = temp

    avg_adv_classifier_acc = None
    avg_adv_classifier_precision = None
    avg_adv_classifier_recall = None

    it = 0

    for epoch in range(n_epochs):

        avg_gen_loss = 0.0
        avg_disc_loss = 0.0
        avg_adv_gen_loss = 0.0

        avg_adv_classifier_acc = 0
        avg_adv_classifier_precision = 0
        avg_adv_classifier_recall = 0

        for batch_idx in range(num_batches):

            inputs, true_labels = dataset.next_batch(args.gpu)

            # Try to craft toxic -> non-toxic adversarial examples
            adversarial_labels = []

            for y_i in true_labels.data.tolist():
                label_i = dataset.idx2label(y_i)

                if label_i == 'toxic':
                    adversarial_labels.append(dataset.label2idx('non_toxic'))
                else:
                    adversarial_labels.append(y_i)

            adversarial_labels = torch.tensor(adversarial_labels)
            adversarial_labels = adversarial_labels.cuda() if args.gpu else adversarial_labels

            # For now, do promotion as well as demotion
            adversarial_labels = 1 - true_labels

            cur_batch_size = inputs.size(1)

            # # Generator training (Eq. 3) using RSGAN loss
            gen_comments = disqus_model.generate_soft_embed(inputs, cur_batch_size)

            disc_scores_real_com_vec = disqus_model.forward_discriminator(inputs.transpose(0, 1).detach())
            disc_scores_gen_com_vec = disqus_model.forward_discriminator_embed(gen_comments.detach())

            disc_scores_real_com = disc_scores_real_com_vec[:, 0]  # Assume Dim 0 is for fake scores (un-normalized)
            disc_scores_gen_com = disc_scores_gen_com_vec[:, 0]
            #
            gen_loss = -torch.mean(F.logsigmoid(disc_scores_real_com - disc_scores_gen_com))

            avg_gen_loss += gen_loss.item()

            gen_loss.backward()
            trainer_G.step()

            trainer_G.zero_grad()
            trainer_D.zero_grad()
            #
            # # Discriminator training (Eq. 3) using RSGAN loss
            gen_comments = disqus_model.generate_soft_embed(inputs, cur_batch_size, temp=new_temp)

            disc_scores_real_com_vec = disqus_model.forward_discriminator(inputs.transpose(0, 1).detach())
            disc_scores_gen_com_vec = disqus_model.forward_discriminator_embed(gen_comments.detach())

            disc_scores_real_com = disc_scores_real_com_vec[:, 0]
            disc_scores_gen_com = disc_scores_gen_com_vec[:, 0]

            disc_loss = -torch.mean(F.logsigmoid(disc_scores_gen_com - disc_scores_real_com))

            avg_disc_loss += disc_loss.item()

            disc_loss.backward()
            trainer_D.step()

            trainer_G.zero_grad()
            trainer_D.zero_grad()

            # Generator training (Eq. 5) using adversarial loss
            gen_comments = disqus_model.generate_soft_embed(inputs, cur_batch_size, temp=new_temp)

            classifier_scores = disqus_model.forward_classifier_embed(gen_comments.detach())
            adv_gen_loss = F.cross_entropy(classifier_scores, adversarial_labels)

            avg_adv_gen_loss += adv_gen_loss.item()

            adv_classifier_acc, adv_classifier_precision, adv_classifier_recall = calc_metrics(classifier_scores,
                                                                                               true_labels)
            avg_adv_classifier_acc += adv_classifier_acc
            avg_adv_classifier_precision += adv_classifier_precision
            avg_adv_classifier_recall += adv_classifier_recall

            adv_gen_loss.backward()
            trainer_G.step()

            trainer_G.zero_grad()
            trainer_D.zero_grad()

            if it % log_interval == 0:
                rand_idx = np.random.randint(0, cur_batch_size)

                x = inputs[:, rand_idx].unsqueeze(dim=1)  # select random sentence from minibatch for sampling
                h_fc = disqus_model.forward_encoder(x)

                sample_idxs = disqus_model.sample_comment(h_fc)

                sample_sent = dataset.idxs2sentence(sample_idxs)
                orig_sent = dataset.idxs2sentence_suppressed(disqus_model, x.squeeze(dim=1))

                print('Iter-{};  Adv_gen_loss: {:.4f}; Adv_classifier_acc: {:.4f}'  # Gen_loss: {:.4f}; Disc_loss: {:.4f};
                      .format(it, adv_gen_loss.item(), adv_classifier_acc)) # gen_loss.item(), disc_loss.item(),
                print('Original sentence: "{}"'.format(orig_sent))
                print('Sampled sentence: "{}"'.format(sample_sent))
                print()

            it += 1

            # Anneal learning rate
            new_lr = lr * (0.5 ** (it // lr_decay_every))
            for param_group_gen in trainer_G.param_groups:
                param_group_gen['lr'] = new_lr

            for param_group_disc in trainer_D.param_groups:
                param_group_disc['lr'] = new_lr

            # Anneal temperature
            new_temp = temp * (0.5 ** (it // temp_decay_every))

        avg_gen_loss /= num_batches
        avg_disc_loss /= num_batches
        avg_adv_gen_loss /= num_batches

        avg_adv_classifier_acc /= num_batches
        avg_adv_classifier_precision /= num_batches
        avg_adv_classifier_recall /= num_batches

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        adv_gen_losses.append(avg_adv_gen_loss)
        adv_classifier_accuracies.append(avg_adv_classifier_acc)

    with open('training_curves/recon_losses.pkl', 'wb') as file:
        pickle.dump(recon_losses, file)

    plt.plot(gen_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Generator loss')
    plt.show()

    plt.plot(disc_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator loss')
    plt.show()

    plt.plot(adv_gen_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Adversarial Gen. loss')
    plt.show()

    plt.plot(adv_classifier_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Adversarial classifier acc.')
    plt.show()

    print('Final adversarial classifier accuracy: ', avg_adv_classifier_acc)
    print('Final adversarial classifier precision: ', avg_adv_classifier_precision)
    print('Final adversarial classifier recall: ', avg_adv_classifier_recall)

    if args.save:
        save_model()

    exit(0)


def save_model():
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    torch.save(disqus_model.state_dict(), 'saved_models/disqus_model.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:

        plt.plot(gen_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Generator loss')
        plt.show()

        plt.plot(disc_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Discriminator loss')
        plt.show()

        plt.plot(adv_gen_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Adversarial Gen. loss')
        plt.show()

        plt.plot(adv_classifier_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Adversarial classifier acc.')
        plt.show()

        if args.save:
            save_model()

        exit(0)

