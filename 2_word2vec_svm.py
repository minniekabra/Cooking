#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:16:25 2018

@author: minnie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:31:20 2018

@author: minnie
"""

from sklearn import svm
import pickle
import gensim
from gensim.models import Word2Vec
import numpy as np
import json
import collections
import csv
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

start = time.clock()

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/train.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)
    
ingredients_list = []
cuisine =[]

for data in json_dict:
    cuisine.append(data['cuisine'])
    ingredients_list.append(data['ingredients'])
    
group_cuisine=dict(collections.Counter(cuisine))

    
#Word2Vec

model_save = Word2Vec.load('model.bin')

#Construct vectors from the word2vec

ingredients_vector = []

for i in range(0, len(ingredients_list)):
    k = 0 
    
    for j in range(0, len(ingredients_list[i])):
        k = k+model_save[ingredients_list[i][j]]/len(ingredients_list[i])
        
    ingredients_vector.append(k)
    
print('data formed')

ingredients_vector_bal = []
cuisine_bal =[]

for i in range(0, len(ingredients_vector)):
    ingredients_vector_bal.append(ingredients_vector[i])
    cuisine_bal.append(cuisine[i])
    
    if cuisine[i] in ('jamaican', 'russian', 'brazilian'):
        for m in range(0,2):
            ingredients_vector_bal.append(ingredients_vector[i])
            cuisine_bal.append(cuisine[i])
            
    if cuisine[i] in ('spanish', 'korean', 'vietnamese', 'moroccan','british', 'filipino', 'irish'):
        for m in range(0,1):
            ingredients_vector_bal.append(ingredients_vector[i])
            cuisine_bal.append(cuisine[i])
        


X = ingredients_vector_bal
Y = cuisine_bal

print('Data size',len(X), len(Y))

group_cuisine=dict(collections.Counter(cuisine_bal))

balancedcuisine = []

for key, value in group_cuisine.items() :
    k = []
    k.append(key)
    k.append(value)
    balancedcuisine.append(k)

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/balancedcuisine.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(balancedcuisine)
    
#One V/S One
'''
clf_SVClinear_databal1 = svm.SVC(C = 1, kernel = 'linear', decision_function_shape = 'ovo')
clf_SVClinear_databal1.fit(X,Y)
clf_SVClinear_databal1 = pickle.dumps(clf_SVClinear_databal1)
'''
clf0 = pickle.loads(clf_SVClinear_databal1)

print('SVClinear ovo Model Trained')

accuracy = 0
mismatch_matrix = balancedcuisine

for i in range(0, len(mismatch_matrix)):
    mismatch_matrix[i][1] = 0
    

italian_mismatch = 0 
mexican_mismatch = 0 
southern_us_mismatch = 0
indian_mismatch = 0
irish_mismatch =0
jamaican_mismatch =0
russian_mismatch =0
brazilian_mismatch =0
greek_mismatch = 0
irish_mismatch = 0

for i in range(0, len(cuisine)):
    k = clf0.predict(ingredients_vector[i])
    
    if k != cuisine[i]:
        #search prediction in mismatch_matrix
        for j in range(0, len(mismatch_matrix)):
            if k == mismatch_matrix[j][0]:
                mismatch_matrix[j][1] = mismatch_matrix[j][1] + 1
                break
        
    if k == cuisine[i]:
        accuracy = accuracy + 1/len(cuisine)
        

print('Accuracy of SVClinear ovo model is ', accuracy)
print('Time Taken by process is' , (time.clock() - start))

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/mismatch_matrix_databal1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['cuisine', 'mismatch'])
    writer.writerows(mismatch_matrix)

'''
#Scoring
with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)
    
testdata_list = []
id_list=[]

for data in json_dict:
    id_list.append(data['id'])
    testdata_list.append(data['ingredients'])
    

#Word2Vec
model_save = Word2Vec.load('model.bin')

#Construct vectors from the word2vec
testdata_vector = []

print('W2V will start')

for i in range(0, len(testdata_list)):
    k = 0 
    
    for j in range(0, len(testdata_list[i])):
        if testdata_list[i][j] in model_save:
            k = k+model_save[testdata_list[i][j]]/len(testdata_list[i])
        
    testdata_vector.append(k)

print('W2V done')

clf0=pickle.loads(clf_SVClinear_databal1)
    
testcuisine = []

for i in range(0, len(id_list)):
    
    k = clf0.predict(testdata_vector[i])
    result = [id_list[i], k[0]]

    testcuisine.append(result)
    
print('Scoring done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_0.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['id', 'cuisine'])
    writer.writerows(testcuisine)
    
with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_0.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print('csv done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all2/kaggle_submission.csv', 'r') as f:
    reader = csv.reader(f)
    list_kaggle = list(reader)
    
for i in range(0,len(list_kaggle)):
    for j in range(0,len(your_list)):
        if list_kaggle[i][0] == your_list[j][0]:
            list_kaggle[i][1] = your_list[j][1]
            break

print('New List')
for i in range(0,10):
    print(list_kaggle[i])

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_kaggle_0.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(list_kaggle)

print('csv done')
'''
