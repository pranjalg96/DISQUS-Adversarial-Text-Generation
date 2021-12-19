import csv
import numpy as np
import torch
from torch.autograd import Variable


# Comments with , are being double quoted. Don't know how to fix it yet.
with open('DISQUS/disqus_dataset/disqus_data_valid.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel', delimiter='"')

    for i, row in enumerate(csvreader):
        print(i, row[1])




