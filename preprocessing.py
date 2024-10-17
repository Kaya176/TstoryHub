import pandas as pd
import re
from tqdm import tqdm

train_data = pd.read_table("ratings_train.txt")
test_data = pd.read_table("ratings_test.txt")

#각 데이터의 갯수 
before_train = len(train_data)
before_test = len(test_data)

#Define text cleaning method
def cleaning(text):
    #text에서 한글을 제외한 모든 문자열을 공백으로 처리해줍니다.
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]"," ",text)
    #공백이 너무 많아지므로 split후 단어와 단어 사이에 공백이 하나만 들어갈 수 있도록 join을 사용해줍니다.
    text = " ".join(text.split())
    return text

print("[train]preprocessing...")
#중복 text를 제거해줍니다.
train_data.drop_duplicates(subset = ["document"],inplace = True)
#nan값을 제거해줍니다. (nan값이 있으면 해당 행 자체를 삭제합니다.)
train_data = train_data.dropna(axis = 0)

#cleaning함수 적용
'''
중복과 nan값을 제거한 뒤 남은 text를 이용하여 위에서 정의한 cleaning함수를 사용하여
한글만 남을 수 있도록 처리해줍니다.
'''
print("[train]cleaning...")
train_data['document'] = [cleaning(t) for t in train_data['document']]

#중복과 nan값을 한번 더 제거해줍니다.
'''
이모티콘 및 숫자로만 이루어진 데이터의 경우, cleaning함수를 적용하게 되면 빈 텍스트만 남게 되고, cleaning을 한 이후에 중복이 있을 수 있으므로 
중복을 제거해준 뒤, 비어있는 값들에 대해서는 nan 처리를 다시 해줍니다.
'''
train_data.drop_duplicates(subset = ['document'],inplace = True)
train_data = train_data.dropna(axis = 0)

print("[train]done!")

#테스트 데이터에 대해서도 훈련데이터에서 했던것과 마찬가지로 같은 작업을 수행해줍니다.
test_data.drop_duplicates(subset=['document'],inplace = True)
test_data = test_data.dropna(axis = 0)
test_data['document'] = [cleaning(t) for t in test_data['document']]
test_data.drop_duplicates(subset = ['document'],inplace = True)
test_data = test_data.dropna(axis = 0)

#전처리 후, 데이터의 갯수 구하기
after_train = len(train_data)
after_test = len(test_data)

#result
print("=== 전처리 전 ===")
print('훈련 데이터의 갯수 : %d  |  테스트 데이터의 갯수 : %d'%(before_train,before_test))
print("=== 전처리 후 ===")
print("훈련 데이터의 갯수 : %d  |  테스트 데이터의 갯수 : %d"%(after_train,after_test))

#save
print("Save...")
train_data.to_csv("new_ratings_train.txt")
test_data.to_csv("new_ratings_test.txt")
print("Done!")