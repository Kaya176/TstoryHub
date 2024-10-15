##작업창 clear
#import os
#os.system("clear")
'''
Top Accuracy : 82%
'''
#warning 지우기
import warnings
warnings.filterwarnings('ignore')

from load_data import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchtext.legacy.data import Iterator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#import wandb
#wandb.init(project='CNN-clf', entity='kaya176')
#config = wandb.config
#config.learning_rate = 0.0002
from knockknock import discord_sender
class PositionalEncoding(nn.Module):
    
    def __init__(self,max_len,d_model,device):
        super(PositionalEncoding,self).__init__()
        
        self.encoding = torch.zeros(max_len,d_model,device = device)
        self.encoding.requires_grad = False # we don't need to compute gradient
        
        pos = torch.arange(0,max_len,device = device)
        pos = pos.float().unsqueeze(dim = 1)
        
        _2i = torch.arange(0,d_model,step = 2,device = device).float()
        
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        
    def forward(self,x):
        #batch_size,seq_len,d_model = x.size()
        seq_len = x.size(1)
        #print(seq_len)
        return self.encoding[:seq_len,:]

class CNN_network(nn.Module):

    def __init__(self,embedding_size,seq_length,device,**kwargs):
        super(CNN_network,self).__init__(**kwargs)
        #embedding layer을 정의. 
        '''
        load_data.py에서 정의한 TEXT field를 이용하여 Embedding(Fasttext) layer를 정의해줍니다.
        '''
        self.seq_length = seq_length #seq_length
        self.embedding_size = embedding_size
        self.kernel = [2,4,5]
        self.output_size = 128
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        #self.position = PositionalEncoding(max_len = seq_length,d_model = embedding_size,device = device)
        #Convolution layer
        self.conv1 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[0],stride=1) #seq_length, out_seq,kernel_size
        self.conv2 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[1],stride=1)
        self.conv3 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[2],stride=1)

        #pooling layer
        self.pool1 = nn.MaxPool1d(self.kernel[0],stride = 1)
        self.pool2 = nn.MaxPool1d(self.kernel[1],stride = 1)
        self.pool3 = nn.MaxPool1d(self.kernel[2],stride = 1)

        #Dropout & FC layer
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(self._calculate_features(),1024)
        #self.linear1 = nn.Linear(9472,1024)
        self.linear2 = nn.Linear(1024,128)
        self.linear3 = nn.Linear(128,1)
    
    def _calculate_features(self):
        '''
        claculates the number of outputs features after Convolution + Maxpooling

        convolved features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) -1 )/ stride ) + 1
        Pooled features = ((embedding_size + (2*padding) - dilation * (kernel - 1) - 1)/stride) + 1

        Source  : https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        out_conv1 = (self.seq_length - 1 * (self.kernel[0] - 1)-1) + 1
        out_conv1 = math.floor(out_conv1)
        out_pool1 = (out_conv1 - 1 * (self.kernel[0]-1)-1 ) + 1
        out_pool1 = math.floor(out_pool1)
        #print(out_pool1)

        out_conv2 = (self.seq_length - 1 * (self.kernel[1] - 1)-1) + 1
        out_conv2 = math.floor(out_conv2)
        out_pool2 = (out_conv2 - 1 * (self.kernel[1]-1)-1 ) + 1
        out_pool2 = math.floor(out_pool2)
        #print(out_pool2)

        out_conv3 = (self.seq_length - 1 * (self.kernel[2] - 1)-1) + 1
        out_conv3 = math.floor(out_conv3)
        out_pool3 = (out_conv3 - 1 * (self.kernel[2]-1)-1 ) + 1
        out_pool3 = math.floor(out_pool3)
        #print(out_pool3)
        out = (out_pool1 + out_pool2 + out_pool3) * 128
        #print(out)
        return out

    def forward(self,input_sentence,size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        '''@
        여기를 참고해보는건 어떨까?
        [1] https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
        [2] https://www.kaggle.com/robertke94/pytorch-textcnn-relu

        --------
        1. 모델 완성해서 validation 데이터셋 구성해서 실험완성하기!
        2. 결과도 어느정도 잘 나오면 LSTM 모델로 바로 바꿔서 실험해보기! (여기까지만 하고 끝내자.)

        --------
        '''
        x = self.embedding(input_sentence)
        #x_position = self.position(input_sentence)
        #x = x + x_position
        #print(x.shape)
        x = x.transpose(1,2)
        #print("? : ",x.shape)
        x1 = self.conv1(x)
        x1 = F.sigmoid(x1)
        x1 = self.pool1(x1)
        

        x2 = self.conv2(x)
        x2 = F.sigmoid(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x)
        x3 = F.sigmoid(x3)
        x3 = self.pool3(x3)

        #print("x1,x2,x3의 사이즈 : ",x1.shape,x2.shape,x3.shape)
        x_concat = torch.cat((x1,x2,x3),2) #2번째 차원 기준으로 묶음.(32,30,17) + (32,30,13) => (32,30,30)
        x_concat = torch.flatten(x_concat,1) #batch를 제외한 나머지를 묶어버린다 -> FC layer를 사용하기 위함. (32,30,30) -> (32,900)
        #print(x_concat.shape)
        out = self.linear1(x_concat)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = F.sigmoid(out)
        
        return out.squeeze()
#@discord_sender(webhook_url = 'https://discord.com/api/webhooks/1219653934847889581/fPgQovXIpua7Dbcy_eqQ5vsPXCibsj7dYil_ZfCOiNANNheljJY37OKmcF75dVG2FRkS')
def train(model,train_data,valid_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #훈련 모드 ON
    model.train()
    #optimizer 정의
    optimizer = optim.AdamW(model.parameters(),lr = 2e-4)
    #loss 출력용
    loss_fn = torch.nn.BCELoss()
    t_loss = 0
    epochs = 5
    #gradient accumulation 적용
    accumulation_step = 4
    #scheduler 적용
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,T_0 = 10,T_mult=2,eta_min=1e-3)
    d_iter = len(train_data)
    for epoch in range(epochs):

        t_loss = 0
        for batch_idx,batch in tqdm(enumerate(train_data)):

            text = batch.document
            label = batch.label

            label = torch.tensor(label,dtype=torch.float)
            size = len(text)
            out = model(text,size)
            loss = loss_fn(out,label) / accumulation_step #새로 추가한 부분
            loss.backward()
            #gradient accumulation 적용
            if not batch_idx % accumulation_step :
                optimizer.step()
                scheduler.step(epoch + batch_idx / d_iter)
                optimizer.zero_grad()
            #update parameters
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #wandb.log({"loss" : loss})
            t_loss += loss.detach().item()

        print(f"Epoch : {epoch + 1} / {epochs} \t Train  Loss : {t_loss/len(train_data) : .3f}")
        test(model,valid_data)

def test(model,data):
    
    #평가 모드로 진입
    model.eval()

    #prediction
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in data:

            text = batch.document
            label = batch.label

            size = len(text)
            y_pred = model(text,size)

            for i in y_pred:
                if i >= 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)

            for j in label:
                labels.append(j.cpu())
    print(f"Accuracy : {accuracy_score(labels,predictions) : .3f}")
    print("sample pred   : ",predictions[:10])
    print('sample labels : ',list(map(int,labels[:10])))
    print("="*100)
        
def predict(model,sentence):
    model.eval()
    with torch.no_grad():
        sent = tokenizer.morphs(sentence)
        sent = torch.tensor([TEXT.vocab.stoi[i] for i in sent])
        sent = F.pad(sent,pad = (1,20-len(sent)-1),value = 1)
        sent = sent.unsqueeze(dim = 0) #for batch
        output = model(sent,len(sent))

        return output.item()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN_network(embedding_size=300,seq_length=20,device = device).to(device)

    train(model,train_batch,valid_batch)
    #test
    print("===TEST===")
    test(model,test_batch)

    while True:
        user = input("테스트 할 리뷰를 작성하세요 : ")
        if user == '0':
            break
        model = model.to('cpu')
        pred = predict(model,user)
        if pred >= 0.5 :
            print(f">>>긍정 리뷰입니다. ({pred : .2f})")
        else:
            print(f">>>부정 리뷰입니다.({pred : .2f})")
    #print("output shape : ",out.shape)
    