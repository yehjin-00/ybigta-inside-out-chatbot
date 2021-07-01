import torch
from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
# from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
# from transformers.modeling_bert import BertModel
from config.FilePathConfig import *
from silence_tensorflow import silence_tensorflow
silence_tensorflow() # tensorflow warning 안 나오게 하기
from models.Modelling import *

class Emotion:
    def __init__(self):
        self.path = CHECKPOINT['EMOTION']
        self.loaded_model = torch.load(self.path, map_location=torch.device('cpu'))

        print('Download BERT ...')
        self.bertmodel, self.vocab = get_pytorch_kobert_model()

        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

        self.model = BERTClassifier(self.bertmodel,  dr_rate=0.5)

        self.max_len = 64

    def predict_number(self, input):
        unseen_test = pd.DataFrame([[input,0]], columns = [['질문 내용','0']])
        unseen_values = unseen_test.values

        test_set = BERTDataset(unseen_values, 0, 1, self.tok, self.max_len, True, False)
        test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=5)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
            token_ids = token_ids.long()
            segment_ids = segment_ids.long()
            valid_length= valid_length
            out = self.loaded_model(token_ids, valid_length, segment_ids)
            out2 = out.cpu().detach().numpy().argmax()
            return str(out2)

    def predict(self, input):
        clf_num = self.predict_number(input)
        image_path = {'0': 'https://github.com/JW12450/Inside_Out_Chatbot/blob/main/images/emotion_anger.gif?raw=true',
                      '1': 'https://github.com/JW12450/Inside_Out_Chatbot/blob/main/images/emotion_sadness.gif?raw=true',
                      '2': 'https://github.com/JW12450/Inside_Out_Chatbot/blob/main/images/emotion_joy.gif?raw=true',
                      '3': 'https://github.com/JW12450/Inside_Out_Chatbot/blob/main/images/emotion_fear.gif?raw=true'}
        return image_path[clf_num]

# 0 -> 버럭이
# 1 -> 슬픔
# 2 -> 기쁨
# 3 -> 소심이