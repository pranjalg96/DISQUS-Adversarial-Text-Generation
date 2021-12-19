import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain


class MODEL_DISQUS(nn.Module):

    def __init__(self, n_vocab, h_dim, z_dim, num_layers_enc=1, bidirectional_enc=False, dropout_enc=0,
                 p_word_dropout=0, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=65,
                 pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(MODEL_DISQUS, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.p_word_dropout = p_word_dropout

        self.num_layers_enc = num_layers_enc
        self.bidirectional_enc_D = 1
        self.dropout_enc = 0

        if bidirectional_enc:
            self.bidirectional_enc_D = 2

        if num_layers_enc > 1:
            self.dropout_enc = dropout_enc

        self.gpu = gpu

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        """
        Encoder is LSTM with FC layer connected to last hidden unit
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, self.num_layers_enc, dropout=self.dropout_enc, bidirectional=bidirectional_enc)
        self.encoder_fc = nn.Linear(self.bidirectional_enc_D*h_dim, h_dim)

        """
        Generator is LSTM with `z` and `x` appended at its inputs
        """
        self.generator = nn.LSTM(self.emb_dim+h_dim+z_dim, h_dim+z_dim)
        self.generator_fc = nn.Linear(h_dim+z_dim, n_vocab)  # Could add an extra FC layer here

        """
        Toxic comment classifier is CNN (for now)
        """
        self.conv3_classifier = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv4_classifier = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv5_classifier = nn.Conv2d(1, 100, (5, self.emb_dim))

        self.disc_fc_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

        self.classifier = nn.ModuleList([
            self.conv3_classifier, self.conv4_classifier, self.conv5_classifier, self.disc_fc_classifier
        ])


        """
        Discriminator is CNN
        """
        self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))

        self.disc_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

        self.discriminator = nn.ModuleList([
            self.conv3, self.conv4, self.conv5, self.disc_fc
        ])

        """
        Grouping the model's parameters: separating encoder and generator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.encoder_fc.parameters()
        )

        self.gen_params = chain(
            self.generator.parameters(), self.generator_fc.parameters()
        )

        self.model_params = chain(
            self.word_emb.parameters(), self.encoder_params, self.gen_params
        )
        self.model_params = filter(lambda p: p.requires_grad, self.model_params)
        self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier.parameters())

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, (h, c) = self.encoder(inputs, None)

        # Forward to generator
        h = h.view(self.bidirectional_enc_D, self.num_layers_enc, -1, self.h_dim)
        h_final = h[:, self.num_layers_enc-1, :, :]  # Only use hidden states from final layer
        h_permuted = h_final.permute(1, 0, 2)
        h_reshaped = h_permuted.reshape(-1, self.bidirectional_enc_D*self.h_dim)
        h_fc = self.encoder_fc(h_reshaped)

        return h_fc

    def sample_noise(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        return z

    def forward_generator(self, inputs, h, z):
        """
        Decoder is a single layer, unidirectional LSTM
        Inputs must be embeddings: seq_len x mbsize
        """

        dec_inputs = self.word_dropout(inputs)

        seq_len = dec_inputs.size(0)
        b_size = dec_inputs.size(1)

        # 1 x mbsize x (z_dim+c_dim)
        init_h = torch.cat([h.unsqueeze(0), z.unsqueeze(0)], dim=2)
        init_c = torch.zeros(1, b_size, self.z_dim+self.h_dim)

        init_c = init_c.cuda() if self.gpu else init_c
        init_h = init_h.cuda() if self.gpu else init_h

        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], dim=2)

        init_h_gen = torch.zeros(1, b_size, self.z_dim+self.h_dim)
        init_h_gen = init_h_gen.cuda() if self.gpu else init_h_gen

        outputs, _ = self.generator(inputs_emb, (init_h_gen, init_c))
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.generator_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def forward_discriminator(self, inputs):
        """
        Inputs is batch of sentences: mbsize x seq_len
        """
        inputs = self.word_emb(inputs)
        return self.forward_discriminator_embed(inputs)

    def forward_discriminator_embed(self, inputs):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.disc_fc(x)

        return y

    def forward_classifier(self, inputs):
        """
        Inputs is batch of sentences: mbsize x seq_len
        """
        inputs = self.word_emb(inputs)
        return self.forward_classifier_embed(inputs)

    def forward_classifier_embed(self, inputs):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim


        x3 = F.relu(self.conv3_classifier(inputs)).squeeze(dim=3)
        x4 = F.relu(self.conv4_classifier(inputs)).squeeze(dim=3)
        x5 = F.relu(self.conv5_classifier(inputs)).squeeze(dim=3)

        # Max-over-time-pool
        # test_tensor = F.max_pool1d(x3, x3.size(2))
        # print(test_tensor.size())
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(dim=2)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(dim=2)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(dim=2)

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.disc_fc_classifier(x)

        return y

    def forward(self, sentence):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of the model
        """
        self.train()

        mbsize = sentence.size(1)

        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        h_fc = self.forward_encoder(enc_inputs)
        z = self.sample_noise(mbsize)

        # Decoder: sentence -> y
        y = self.forward_generator(dec_inputs, h_fc, z)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), reduction='mean'
        )

        return recon_loss

    # def generate_sentences(self, batch_size):
    #     """
    #     Generate sentences and corresponding z of (batch_size x max_sent_len)
    #     """
    #     samples = []
    #     cs = []
    #
    #     for _ in range(batch_size):
    #         z = self.sample_z_prior(1)
    #         c = self.sample_c_prior(1)
    #         samples.append(self.sample_sentence(z, c, raw=True))
    #         cs.append(c.long())
    #
    #     X_gen = torch.cat(samples, dim=0)
    #     c_gen = torch.cat(cs, dim=0)
    #
    #     return X_gen, c_gen

    def sample_comment(self, h_fc, raw=False, temp=1):  # Basically a language model sampling
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z = self.sample_noise(1)

        h_fc, z = h_fc.view(1, 1, -1), z.view(1, 1, -1)

        init_h = torch.cat([h_fc, z], dim=2)
        init_c = torch.zeros((1, 1, self.h_dim+self.z_dim))
        init_c = init_c.cuda() if self.gpu else init_c

        init_h_gen = torch.zeros((1, 1, self.h_dim+self.z_dim))
        init_h_gen = init_h_gen.cuda() if self.gpu else init_h_gen

        if not isinstance(init_h, Variable):
            init_h = Variable(init_h)
            init_c = Variable(init_c)

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, init_h], 2)

            output, (init_h_gen, init_c) = self.generator(emb, (init_h_gen, init_c))
            y = self.generator_fc(output).view(-1)
            y = F.softmax(y/temp, dim=0)

            idx = torch.multinomial(y, 1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, inputs, mbsize, temp=1):
        """
        Generate soft embeddings of (mbsize x max_sent_len x emb_dim)
        """
        samples = []

        for batch_i in range(mbsize):
            x = inputs[:, batch_i].unsqueeze(dim=1)
            h_fc = self.forward_encoder(x)

            samples.append(self.sample_soft_embed(h_fc, temp=temp))

        X_gen = torch.cat(samples, dim=0)

        return X_gen

    def sample_soft_embed(self, h_fc, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z = self.sample_noise(1)
        z = z.cuda() if self.gpu else z

        h_fc, z = h_fc.view(1, 1, -1), z.view(1, 1, -1)

        init_c = torch.zeros((1, 1, self.h_dim + self.z_dim))
        init_c = init_c.cuda() if self.gpu else init_c

        init_h_gen = torch.zeros((1, 1, self.h_dim + self.z_dim))
        init_h_gen = init_h_gen.cuda() if self.gpu else init_h_gen

        if not isinstance(init_c, Variable):
            init_c = Variable(init_c)
            init_h_gen = Variable(init_h_gen)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, h_fc, z], 2)

        outputs = [self.word_emb(word).view(1, -1)]

        for i in range(self.MAX_SENT_LEN):
            output, (init_h_gen, init_c) = self.generator(emb, (init_h_gen, init_c))
            o = self.generator_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with h_fc and z for the next input
            emb = torch.cat([emb, h_fc, z], 2)


        # 1 x max_sent_len x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
