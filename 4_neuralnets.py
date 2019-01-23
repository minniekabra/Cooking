#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:31:38 2019

@author: minnie
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
from keras.layers.core import Dense,Dropout,Activation,Flatten,Lambda
from keras import backend as K
import keras

import pickle
import gensim
from gensim.models import Word2Vec
import json
import collections
import csv
import time
import warnings
import statistics
warnings.filterwarnings("ignore", category=DeprecationWarning)

start = time.clock()

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/train.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)

cuisine = []
ingredients_all = []
ingredients_data = []
id_list = []

for data in json_dict:
    id_list.append(data['id'])
    cuisine.append(data['cuisine'])
    ingredients_data.append(data['ingredients'])
    ingredients_all.extend(data['ingredients'])

groups_cuisine =dict(collections.Counter(cuisine))
cuisine_strength = []

for key, value in groups_cuisine.items():
    k = [key, value]
    cuisine_strength.append(k)

cuisine_all = []    
for i in range(len(cuisine_strength)):
    k = [cuisine_strength[i][0], []]
    cuisine_all.append(k)


for data in json_dict:
    k = data['cuisine']
    
    for j in range(len(cuisine_all)):
        if k == cuisine_all[j][0]:
            cuisine_all[j][1].extend(data['ingredients'])

cuisine_all_prop = cuisine_all


for i in range(len(cuisine_all)):
    cuisine_all_prop[i][0] = cuisine_all[i][0]
    cuisine_all_prop[i][1] = dict(collections.Counter(cuisine_all[i][1]))
    

groups_ingredients = dict(collections.Counter(ingredients_all))
ingredients_list =[]
ingredients_sum  =[]

for key, value in groups_ingredients.items():
    ingredients_list.append(key)
    ingredients_sum.append(value)


ingredients_contents = []
for i in range(len(ingredients_list)):
    k = [(ingredients_list[i]).encode('utf-8')]
    A = [0] * len(cuisine_strength)
    k.extend(A)
    ingredients_contents.append(k)

#Important check


for i in range(0, len(ingredients_contents)):
    ingr =  ingredients_list[i]
    
    for key, value  in groups_ingredients.items():
        
        den1 = 1
        if key == ingr:
            den1 = value
            break

    for j in range(1, (len(cuisine_all_prop)+ 1)):
        n1 = 0
        n2 = 0
        cuis = cuisine_all_prop[j-1][0]            
        den2 = cuisine_strength[j-1][1]
            
        for key, value in cuisine_all_prop[j-1][1].items():
            if key == ingr:
                n2 = value
            
        num = max(n1, n2) 
        no = (num/(den1*den2))*40000
        ingredients_contents[i][j] = no
                

        
header = ['ingredient']

for i in range(len(cuisine_strength)):
    ingr = cuisine_strength[i][0].encode('utf-8')
    header.append(ingr)
    
    
with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/enhance2/ingredients_content_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(ingredients_contents)

ingr_variance = []

for i in range(0, len(ingredients_contents)):
    
    std = statistics.stdev(map(float, ingredients_contents[i][1:len(ingredients_contents[i])]))
    avg = statistics.mean(map(float, ingredients_contents[i][1:len(ingredients_contents[i])]))
    coeff_varn = std/avg
    k =[std, avg, coeff_varn]
    ingr_variance.append(k)
    


ingredients_contents_2 = []

for i in range(0, len(ingredients_contents)):
    k = [ingredients_contents[i][0]]
    for j in range(1, len(ingredients_contents[i])):
        var = ingredients_contents[i][j]*ingr_variance[i][2]
        k.append(var)
        
    ingredients_contents_2.append(k)
    

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/enhance2/ingredients_content_2.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(ingredients_contents_2)   

#Data formation

ingredients_vector = []
t1 = 0
t2 = 0
t3 = 0
z = 0
Y = []

for i in range(0, len(ingredients_data)):

    initial_list = [0]*len(cuisine_strength)
    t1 = t1+1
    #Y.append(cuisine[i])
    
    for j in range(len(ingredients_data[i])):
        ingr= ingredients_data[i][j].encode('utf-8')
        t2 = t2 + 1
        
        for k in range(len(ingredients_contents_2)):
            t3 = t3+1
           
            if ingredients_contents_2[k][0] == ingr:
                z = z+1
                initial_list = [(x + y) for x, y in zip(initial_list, ingredients_contents_2[k][1: len(ingredients_contents_2[k])])]
                
        
    no_ingr = len(ingredients_data[i])
    initial_list.append(no_ingr)
    initial_list.append(cuisine[i])
    ingredients_vector.append(initial_list)
    
  

print('Time Taken in data formation' , (time.clock() - start)/60, 'mins')


with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/ingredients_vector.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(ingredients_vector)


seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/ingredients_vector.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:21].astype(float)
Y = dataset[:,21]


print('Input dimesions', len(X), len(X[0]))


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

label_map =[]

for i in range(0, len(dummy_y)):
    actual = Y[i]
    for j in range(0, len(dummy_y[i])):
        if dummy_y[i][j] == 1:
            break
    app = tuple([j, actual])
    label_map.append(app)
    
label_map = list(set(label_map))
label_map = sorted(label_map,key=lambda x: x[0])


print(label_map)
    


print('length of dummy predictor/ No. of classes are',len(dummy_y), len(dummy_y[0]))

#No. of nodes in hidden layer are 500

def baseline_model():
    
	# create model
    model = Sequential()
    #model.add(Lambda(lambda x: K.tf.nn.softmax(x, dim= axis )))
    model.add(Dense(500, input_dim=21, activation='relu'))
    model.add(Dense(20))
    model.add(Activation('softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Batch size as 1000 and no. of epochs as 400
estimator = KerasClassifier(build_fn=baseline_model, epochs=400, batch_size=1000, verbose=0)
estimator.fit(X, dummy_y, epochs=400, batch_size=1000, verbose=0)

scores = estimator.predict(X)
accuracy = 0

for i in range(0, 100):
    if label_map[scores[i]][1] == Y[i]:
        accuracy = accuracy + 1/len(Y)
    
print('Model accuracy on training data', accuracy)

'''
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

print('Time Taken in model' , (time.clock() - start)/60, 'mins')


#Scoring

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)
    
testdata_list = []
id_list=[]

for data in json_dict:
    id_list.append(data['id'])
    testdata_list.append(data['ingredients'])
    
test_vector = []


for i in range(0, len(testdata_list)):

    initial_list = [0]*len(cuisine_strength)
    t1 = t1+1
    
    for j in range(len(testdata_list[i])):
        ingr= testdata_list[i][j].encode('utf-8')
        t2 = t2 + 1
        
        for k in range(len(ingredients_contents_2)):
            t3 = t3+1
           
            if ingredients_contents_2[k][0] == ingr:
                z = z+1
                initial_list = [(x + y) for x, y in zip(initial_list, ingredients_contents_2[k][1: len(ingredients_contents_2[k])])]
                
        
    no_ingr = len(testdata_list[i])
    initial_list.append(no_ingr)
    initial_list.append(0)
    test_vector.append(initial_list)

with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/ingredients_vector_test.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(test_vector)

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/ingredients_vector_test.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:21].astype(float)
Y = dataset[:,21] 

print('length of testdata is', len(X), len(X[0]))
    
testcuisine = []

scores = estimator.predict(X)
for i in range(0, len(test_vector)):
    result = [id_list[i], label_map[scores[i]][1]]
    testcuisine.append(result)
    
print('Scoring done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/NNtest_20_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['id', 'cuisine'])
    writer.writerows(testcuisine)
    
with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/NNtest_20_1.csv', 'r') as f:
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


with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/NN_Keras/NNtest_20_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(list_kaggle)

print('Time Taken by process is' , (time.clock() - start)/60, 'mins')

