import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from torchtext.legacy import data
import torch
import torchtext
'''
torchtext의 Field를 이용하여 훈련 및 테스트에 사용할 데이터를 만들어보도록 하겠습니다.
데이터를 원하는 batchsize에 나누기 전, 전처리한 데이터를 이용하여 형태소 분석을 진행합니다.
'''
#Part 1. Tokenize

#Tokenizer로 사용할 Mecab 객체를 정의합니다.(Okt등 다른 형태소 분석기를 사용해도 됩니다.)
tokenizer = Mecab(dicpath="C:\mecab\mecab-ko-dic")
#stopword(불용어)를 정의합니다. 사용자에 따라서 추가해서 사용할 수 있습니다.
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

#형태소 분석 후에 사용할 처리들을 모아둔 preprocess라는 이름의 함수를 정의합니다.
def preprocess(text):
    #stopword를 제거합니다.
    word = [t for t in text if t not in stopwords]
    #길이가 1이하인 단어들은 제거합니다.
    word = [t for t in word if  len(t) > 1]
    return word
'''
def label_onehot(label):
    onehot = np.zeros(2)
    onehot[int(label)] = 1
    return onehot
'''
#Part 2. Define Field
print("+"*50)
print("load data...")
print("+"*50)
#사용안할 예정
IDX = data.Field(sequential = False, use_vocab = False)
ID = data.Field(sequential= False, use_vocab = False)
#사용할 예정
TEXT = data.Field(fix_length = 20, sequential = True, batch_first = True,is_target = False, use_vocab = True, tokenize = tokenizer.morphs, preprocessing = preprocess)
LABEL = data.Field(sequential = False,batch_first = True,is_target = True,use_vocab = False,dtype = torch.float32)

field = [("idx",IDX),('id',ID),('document',TEXT),('label',LABEL)]

#이전에 처리한 문서를 불러와서 훈련에 사용할 데이터로 만들어줍니다.
train_data, valid_data,test_data = data.TabularDataset.splits(
    path = '.', #반드시 있어야함!
    train = 'new_ratings_train.txt',validation= 'sample_valid.txt' ,test = 'new_ratings_test.txt',
    format = 'csv',
    fields = field,
    skip_header = True
)

print("Done!")
print("+"*50)
print("Samples...")
print("+"*50)
for i in range(5):
    print(vars(train_data[i]))
print("+"*50)

#Part 3. Make data
'''
위의 과정을 거친 data들을 batch 단위로 만들어주며,모델에 입력 할 수 있도록 Embedding 작업을 진행합니다.
Embedding은 사전 훈련된 Fasttext를 사용하며, 모델은 아래 주소에서 다운받을 수 있습니다.

https://fasttext.cc/docs/en/crawl-vectors.html
-bin파일 : fasttext 모델도 같이 들어있는 파일. + Embedding file
-text파일 : Embedding file

물론 다른 모델을 사용해도 되며, Tf-Idf를 적용해보는것도 좋습니다.
'''

vector = torchtext.vocab.Vectors(name = './cc.ko.300.vec')
TEXT.build_vocab(train_data,vectors = vector)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Batch size에 맞게 데이터를 만들어줍니다.
train_batch = data.BucketIterator(
    dataset = train_data,
    sort = False,
    batch_size = 64,
    repeat = False,
    shuffle = True,
    device = device)

valid_batch = data.BucketIterator(
    dataset = valid_data,
    sort = False,
    batch_size = 64,
    repeat = False,
    shuffle = False,
    device = device)

test_batch = data.BucketIterator(
    dataset = test_data,
    sort = False,
    batch_size = 64,
    repeat = False,
    shuffle = False,
    device = device)

print("load data... Done!!")
print("+"*50)