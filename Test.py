# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:56:17 2022

@author: Niousha
"""
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from RUM_NN import Beta, Formula,RUM_NN, gumbel
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from random import seed
seed(1)
# import tensorflow.compat.v1.keras.backend as K
import numpy as np 

epochs=20
test = RUM_NN(iternum=2000, epochs=epochs, batch =500)

Dataset = pd.read_excel('Dataset_MNL.xlsx')

Dataset.drop('Unnamed: 0', inplace=True, axis=1)
target = test.dense_to_one_hot(Dataset['choice'])
globals().update(dict(Dataset))



W1 = Beta('w1', 0, 0)

W2 = Beta('w2', 0, 0)

W3 = Beta('w3', -1, 0)

W4 = Beta('w4', 1, 0)


U2 = Formula([(W1, a2), (W2, b2), (W3, p2), (W4, q2)])

U1 = Formula([(W1, a1), (W2, b1), (W3, p1), (W4, q1)])

U3 = Formula([( W1, a3), (W2, b3), (W3, p3) , (W4, q3)],errorWeight=1)

v = {'1': U2, '0': U1 , '2':U3}



test.create_model(formulaDict=v, errorDist=gumbel, correlation=False, errorLoc=0, errorScale=1, gamma=1e4)

history, model= test.fit_model(target)

model2 = model
test.plot_parameters_history()
test.summarise_history_accuracy()
test.summarise_history_loss()









