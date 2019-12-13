# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,LSTM,Embedding
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
import numpy as np


from django.db import models


class NLP_sets(models.Model):
    name=models.CharField(blank=False,max_length=253)
    Q_targets=models.IntegerField(default=0,blank=False)
    text=models.CharField(default='',max_length=65535)
    category=models.CharField(default='',max_length=253)


    def preprocessing(self,text):
        stop_words_esp=['para','a','la','el','las','ellas','por','y','o','de']
        text=re.sub('[^a-zA-Z-0-9\s]',' ',text).lower()
        return text


    def __str__(self):
        return self.name



class keras_model(models.Model):
    name=models.CharField(max_length=253,blank=False)
    accuracy=models.FloatField(default=0,blank=False)
    active=models.BooleanField(default=False)
    ruta=models.CharField(max_length=253,blank=True)
    model=None

    def entrenar(self):
        datos=NLP_sets.objects.all()

        x_train,x_test,y_train,y_test=train_test_split()
    def __str__(self):
        return self.name

