Original file is located at
    https://colab.research.google.com/drive/1kV--Xm87jUGl8bsIYOn_c8GbmKtxg9mf

<!-- run the following commands on local machine -->
run with
pip install torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
pip install tqdm
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
pip install torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
pip install numpy
import numpy as np
pip install pandas
import pandas as pd
from tqdm import tqdm
pip install sklearn.metrics
from sklearn.metrics import f1_score
pip install codecs
import codecs


<!-- The folder all the files in .conllu format used for training and testing the model, the file locations may change on different machines so the withopen commands should be changed accordingly. -->
