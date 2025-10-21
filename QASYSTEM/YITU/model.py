
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
from transformers import BertModel

from transformers import logging
logging.set_verbosity_error()

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name ,param in self.bert.named_parameters():

            param.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])

        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)#卷积
        out = F.relu(out)#激活
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze(-1).squeeze(-1)#最大池化+压缩最后两个维度
    '''前向传播'''
    def forward(self, input, mask):
        out = self.bert(input, mask)[0].unsqueeze(1)#获取bert最后一层隐藏状态并增加一个维度[B,L,D]--[B,1,L,D]
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        return torch.sigmoid(self.linear(out))#sigmoid激活


if __name__ == '__main__':
    model = TextCNN()
    input = torch.randint(0, 3000, (2, TEXT_LEN))
    mask = torch.ones_like(input)
    print(model(input, mask))

