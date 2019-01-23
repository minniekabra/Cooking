#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:11:51 2018

@author: minnie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:03:28 2018

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
import statistics
warnings.filterwarnings("ignore", category=DeprecationWarning)

start = time.clock()

A = np.array([1,2,3,4])
print(A/10)
print(statistics.mean(map(float, A)))
print(statistics.stdev(A))


with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/train.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)

cuisine = []
ingredients_all = []
ingredients_list = []
id_list = []

for data in json_dict:
    id_list.append(data['id'])
    cuisine.append(data['cuisine'])
    ingredients_list.append(data['ingredients'])
    ingredients_all.extend(data['ingredients'])

groups_cuisine =dict(collections.Counter(cuisine))
groups_ingredients = dict(collections.Counter(ingredients_all))

cuisine_list = []

for key, value in groups_cuisine.items():
    cuisine_list.append(key)
    
cuisine_all =[]
cuisine_ingredients = []
    
for i in range(len(cuisine_list)):
    k = [cuisine_list[i], []]
    m = [cuisine_list[i]]
    cuisine_all.append(k)
    cuisine_ingredients.append(m)
    
print(cuisine_ingredients)
    

for data in json_dict: 
    k = data['cuisine']
    for j in range(len(cuisine_all)):
        if k == cuisine_all[j][0]:
            cuisine_all[j][1].extend(data['ingredients'])
            
cuisine_all_prop =cuisine_all

for i in range(len(cuisine_all)):
   cuisine_all_prop[i][0] = cuisine_all[i][0]
   cuisine_all_prop[i][1] = dict(collections.Counter(cuisine_all[i][1]))

ingredients_percuisine_1 = []
cuisine_1 = []

for i in range(0, len(cuisine_all_prop)):
    A = []
    B =[]
    
    for key, value in cuisine_all_prop[i][1].items():
        A.append(key)
        B.append(value)
    
    C = [A, B]
    ingredients_percuisine_1.append(C)
    print(len(A), len(B), len(C))
    
    cuisine_1.append(cuisine_all_prop[i][0])
    
ingredients_distinct = []
ingredients_percuisine_2 = []

for key, value in groups_ingredients.items():
      
    A = [key.encode('utf-8')]
    x = 0
    for i in range(len(ingredients_percuisine_1)):
        x = 0
        for j in range(0, len(ingredients_percuisine_1[i][0])):
            if key == ingredients_percuisine_1[i][0][j]:
                x = ingredients_percuisine_1[i][1][j]
                break
        A.append(x)
        
    ingredients_percuisine_2.append(A)
    
#write to csv 
header = ['ingredients']
header.extend(cuisine_1)

print(len(ingredients_percuisine_2[0]), ingredients_percuisine_2[0], len(ingredients_percuisine_2))

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/enhance/ingredients_content.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(ingredients_percuisine_2)


ingredients_vector = []

for i in range(0, len(ingredients_list)):

    initial_list = [0]*len(cuisine_list)
    
        
    for j in range(0, len(ingredients_list[i])):
        xyz = ingredients_list[i][j]
        stats =[]
    
        
        for l in range(len(cuisine_all_prop)):
            
            if xyz in cuisine_all_prop[l][1]:
                no_cuisine = groups_cuisine[cuisine_all_prop[l][0]]
                k = ((cuisine_all_prop[l][1][xyz]/groups_ingredients[xyz])/no_cuisine)*40000
                initial_list[l] = initial_list[l] + k
                #initial_list[l] = initial_list[l] + cuisine_all_prop[l][1][xyz]/groups_ingredients[xyz]  
 
            else:
                k = 0
                
            
            stats.append(k)
        
        #mean = statistics.mean(map(float, stats))
        #std = statistics.stdev(map(float, stats))
        
    no_ingr = len(ingredients_list[i])
    initial_list.append(no_ingr)
    
    for k in range(len(cuisine_ingredients)):
        if cuisine_ingredients[k][0] == cuisine[i]:
            cuisine_ingredients[k].append(initial_list)
            break
        
    
    ingredients_vector.append(initial_list)
    
#print(len(ingredients_vector[0]),ingredients_vector[0] )

avg_cuisine_ingredients = []

for i in range(len(cuisine_ingredients)):    
    comb = [0]*21    
    for j in range(1, len(cuisine_ingredients[i])):
        first = comb
        second = cuisine_ingredients[i][j]
        comb = list(map(sum, zip(first, second)))
    
    
    comb = np.array(comb)/(len(cuisine_ingredients[i]) - 1)
    k = [cuisine_ingredients[i][0]]
    k.extend(list(comb))
    avg_cuisine_ingredients.append(k)
    
X = ingredients_vector
Y = cuisine

header = ['cuisine']
for i in range(0, len(cuisine_list)):
    header.append(cuisine_list[i])
    
header.append('no_ingr')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/enhance/avg_cuisine.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(avg_cuisine_ingredients)

print('Data length', len(X), len(X[0]), len(Y))
clf_SVClinear_ub20 = svm.SVC(C = 1, kernel = 'linear', decision_function_shape = 'ovo')
clf_SVClinear_ub20.fit(X,Y)
clf_SVClinear_ub20 = pickle.dumps(clf_SVClinear_ub20)

print('Modelformed')

clf = pickle.loads(clf_SVClinear_ub20)
accuracy =0
matrix = []

for i in range(0, len(cuisine)):
    k = clf.predict(ingredients_vector[i])
    
        
    if k == cuisine[i]:
        accuracy = accuracy + 1/len(cuisine)
        mismatch =0
    else:
        mismatch = 1
        
        
    A =[]
    A.append(id_list[i])
    A.append(cuisine[i])
    A.append(str(k).encode(encoding = 'utf-8'))
    A.append(mismatch)
    A.extend(ingredients_vector[i])
    matrix.append(A) 
    
print('Accuracy of SVC linear ovo model is ', accuracy)

#write to csv

header = ['id', 'cuisine', 'predicted', 'mismatch']

for i in range(0, len(cuisine_list)):
    header.append(cuisine_list[i])

header.append('no_ingr')
    
print(header, len(header))

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/enhance/train_scoring_ub20.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(matrix)

'''
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
    
    initial_list = [0]*len(cuisine_list)
    
    for j in range(0, len(testdata_list[i])):
        xyz =testdata_list[i][j]
            
        for l in range(len(cuisine_all_prop)):
            if xyz in cuisine_all_prop[l][1]:
                initial_list[l] = initial_list[l] + cuisine_all_prop[l][1][xyz]/groups_ingredients[xyz]
        
    

    testdata_vector.append(initial_list)
        
    #testdata_vector.append(k)
    

print('W2V done', len(testdata_vector[0]))

clf = pickle.loads(clf_SVClinear_ub20)
    
testcuisine = []
mismatch_matrix = []

for i in range(0, len(testdata_vector)):
    
    k = clf.predict(testdata_vector[i])
    result = [id_list[i], k[0]]

    testcuisine.append(result)
    
print('Scoring done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_20_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['id', 'cuisine'])
    writer.writerows(testcuisine)
    
with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_20_1.csv', 'r') as f:
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


with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test_kaggle_20.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(list_kaggle)
'''
 
print('Time Taken by process is' , (time.clock() - start))


    