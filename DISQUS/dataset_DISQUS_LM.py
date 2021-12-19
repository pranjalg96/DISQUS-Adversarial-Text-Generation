import torchtext

from torchtext.legacy import data, datasets
from torchtext.vocab import GloVe

import csv
import pickle


class DATA_DISQUS_LM:

    def __init__(self, batch_size=32, emb_dim=300):
        self.TEXT_vocab = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=65)

        self.train_data = data.TabularDataset(
            path='DISQUS/disqus_dataset/disqus_data_filtered.csv',
            format='csv',
            fields=[('text', self.TEXT_vocab)],
            csv_reader_params={'dialect': 'excel', 'delimiter': '"'},
            skip_header=False
        )

        self.TEXT_vocab.build_vocab(self.train_data, vectors=GloVe('6B', dim=emb_dim))

        self.n_vocab = len(self.TEXT_vocab.vocab.itos)
        print('Size of vocabulary: ', self.n_vocab)

        self.emb_dim = emb_dim
        self.batch_size = batch_size

        with open('built_vocab.pkl', 'wb') as file:
            pickle.dump(self.TEXT_vocab.vocab, file)

    def get_vocab_vectors(self):
        return self.TEXT_vocab.vocab.vectors

    def create_batch_iterable(self):

        self.train_iter = data.BucketIterator(
            self.train_data, batch_size=self.batch_size, device='cuda',
            shuffle=True, repeat=True  # Should I set repeat to false?
        )
        self.train_iter = iter(self.train_iter)

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda()

        return batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT_vocab.vocab.itos[i] for i in idxs])

    def idxs2sentence_suppressed(self, model, idxs):
        """
        Suppress the pad, start and eos tokens while displaying sentence
        """
        display_list = []

        for i in idxs:

            if i == model.PAD_IDX or i == model.START_IDX or i == model.EOS_IDX:
                continue

            display_list.append(i)

        return ' '.join([self.TEXT_vocab.vocab.itos[i] for i in display_list])

    # def idx2label(self, idx):
    #     return self.LABEL_8k.vocab.itos[idx]


