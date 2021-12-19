import csv
import numpy as np
import torch
from torch.autograd import Variable

recover_unsuper = []

# Comments with , are being double quoted. Don't know how to fix it yet.
with open('DISQUS/disqus_dataset/disqus_data_filtered.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel', delimiter='"')

    for row in csvreader:
        print(row)




