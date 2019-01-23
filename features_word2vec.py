#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:52:48 2018

@author: minnie
"""
import gensim
from gensim.models import Word2Vec

import json
import collections

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/train.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)
    
ingredients_list = []
cuisine =[]

for data in json_dict:
    cuisine.append(data['cuisine'])
    ingredients_list.append(data['ingredients'])
    
print(len(cuisine))
print(len(ingredients_list))

#Word2Vec
'''
model = gensim.models.Word2Vec(ingredients_list, size = 500, window = 10, min_count = 1, workers = 10)
print('Initialized W2V')
model.train(ingredients_list, total_examples=len(ingredients_list), epochs=10)
print('Trained W2V', model)
model.save('model.bin')
'''
model_save = Word2Vec.load('model.bin')
#print(len(model_save['wheat']))
#print(model_save['wheat'])
#print(model_save.wv.most_similar(positive ='wheat flour'))
#print(model_save.wv.most_similar(positive ='seasoning'))
print(model_save.wv.similarity('salt', 'water'))
print(model_save.wv.similarity('wheat flour', 'salt'))
print(model_save.wv.similarity('wheat flour', 'warm water'))
print(model_save.wv.similarity('tumeric', 'onions'))

#Construct vectors from the word2vec

ingredients_vector = []
ing = [0]*500

for i in range(0, len(ingredients_list)):
    k = 0 
    
    for j in range(0, len(ingredients_list[i])):
        k = k+model_save[ingredients_list[i][j]]/len(ingredients_list[i])   
        
    ingredients_vector.append(k)

k = 0
similarity_indian = []

for z in range(0, len(cuisine)):
    if cuisine[z] == 'indian' and k == 0:
        k = 1
        arr1 = ingredients_vector[z]
    
    if cuisine[z] == 'indian' and k == 1:
        arr2 = ingredients_vector[z]
        similarity = sum(x * y for x, y in zip(arr1, arr2))
        similarity_indian.append(similarity)
        arr1 = arr2
        
        
print('Similarity within India is ',sum(similarity_indian)/len(similarity_indian))

#Mexican & Indian

k = 0
similarity_mex_indian = []
arr1 = []
arr2 = []

for z in range(0, len(cuisine)):
    k = 0
    if cuisine[z] == 'indian':
        arr1 = ingredients_vector[z]
        k = 1
    
    if cuisine[z] == 'mexican':
        arr2 = ingredients_vector[z]
        k = 1
    
    if k == 1 and len(arr1)>0 and len(arr2) > 0:
        similarity = sum(x * y for x, y in zip(arr1, arr2))
        similarity_mex_indian.append(similarity)
    
print('Similarity b/w India & Mexico ', sum(similarity_mex_indian)/len(similarity_mex_indian))


k = 0
similarity_french_indian = []
arr1 = []
arr2 = []

for z in range(0, len(cuisine)):
    k = 0
    if cuisine[z] == 'indian':
        arr1 = ingredients_vector[z]
        k = 1
    
    if cuisine[z] == 'french':
        arr2 = ingredients_vector[z]
        k = 1
    
    if k == 1 and len(arr1)>0 and len(arr2) > 0:
        similarity = sum(x * y for x, y in zip(arr1, arr2))
        similarity_french_indian.append(similarity)
    
print('Similarity b/w India & France ', sum(similarity_french_indian)/len(similarity_french_indian))

k = 0
similarity_french_spanish = []
arr1 = []
arr2 = []

for z in range(0, len(cuisine)):
    k = 0
    if cuisine[z] == 'italian':
        arr1 = ingredients_vector[z]
        k = 1
    
    if cuisine[z] == 'french':
        arr2 = ingredients_vector[z]
        k = 1
    
    if k == 1 and len(arr1)>0 and len(arr2) > 0:
        similarity = sum(x * y for x, y in zip(arr1, arr2))
        similarity_french_spanish.append(similarity)
    
print('Similarity b/w France & Italy ', sum(similarity_french_spanish)/len(similarity_french_spanish))