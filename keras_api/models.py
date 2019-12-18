# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,LSTM,Embedding,Input,Bidirectional
from keras.callbacks import ReduceLROnPlateau
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



    def __str__(self):
        return self.name



class keras_model(models.Model):
    name=models.CharField(max_length=253,blank=False)
    accuracy=models.FloatField(default=0,blank=False)
    active=models.BooleanField(default=False)
    ruta=models.CharField(max_length=253,blank=True)
    model=None

    def procesar(self):
        # Obtener datos del objeto NLP SETS preprocesarlos y adjuntarlos en un diccionario

        datos = NLP_sets.objects.all()
        df = {"texto": [], "category": []}

        max_len = 0
        stop_words_esp = ['para', 'a', 'la', 'el', 'las', 'ellas', 'por', 'y', 'o', 'de']
        for i in datos:

            p = len(i.text.split())
            if p > max_len:
                max_len = p
        if max_len < 300:

            for i in datos:
                text = i.text


                text = re.sub('[^a-zA-Z-0-9\s]', ' ', text).lower()
                text=' '.join([x for x in text.split() if x not in stop_words_esp])
                df['texto'].append(text)
                df['category'].append(i.category)
        else:
            import random
            random.seed(55)

            for i in datos:
                texto = i.text
                text = re.sub('[^a-zA-Z-0-9\s]', ' ', texto).lower()
                text = ' '.join([x for x in text.split() if x not in stop_words_esp])
                df['texto'].append(text)
                df['category'].append(i.category)

            X = df['texto']
            nuevo_X = []
            print("preprocesando los datos. . . ")
            a = 0
            for i in X[:]:

                print("procesando avisos :: {} ../..{}".format(a, len(X)))
                count = 0
                indic = []
                for c in range(len(i)):
                    if len(i) <= 2500:

                        while count <= len(i) / 10 and len(i) > 250:
                            redi = random.randint(0, len(i))
                            if redi == len(i):

                                indic.append(i[redi - 1])
                            else:
                                indic.append(i[redi])
                            count += 1

                    elif len(i) >= 2501 and len(i) <= 10000:
                        while count <= len(i) / 33:
                            redi = random.randint(0, len(i))
                            if redi == len(i):

                                indic.append(i[redi - 1])
                            else:
                                indic.append(i[redi])
                            count += 1
                            count += 1

                    elif len(i) > 10000:

                        while count <= len(i) / 300:
                            redi = random.randint(0, len(i))
                            if redi == len(i):

                                indic.append(i[redi - 1])
                            else:
                                indic.append(i[redi])
                            count += 1
                            count += 1

                if bool(indic):

                    nuevo_X.append(indic)
                    a += 1
                else:
                    nuevo_X.append(i)
                    a += 1
            df['texto'] = nuevo_X
            maximo = max([len(x) for x in nuevo_X])
            print(maximo)

        # aplicar labelencoder para generar rank 1 array, 1D para clasificacion obtenemos Y

        print("preparando dummies spliteando los datos , tokenizando sequencias . . . ")
        lab = LabelEncoder()
        lab.fit(df['category'])
        y = lab.transform(df['category'])

        # generamos el inverse transform de las labelizaciones para crear "clases" el diccionario que nos va a ayudar a mostrar las clases predichas
        abc = []
        for i in lab.inverse_transform(y):
            abc.append(i)

        clases = {}
        for i in range(len(abc)):
            clases[y[i]] = abc[i]

        bow = set()
        for i in df['texto']:
            for k in i:
                bow.add(k)
        maxim = len(bow)

        # tokenizamos las palabras utilizando tokenizer de keras.preprocessing.text

        token = Tokenizer(num_words=maxim)
        token.fit_on_texts(df['texto'])
        x = token.texts_to_sequences(df['texto'])
        x = pad_sequences(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=55)

        return x_train,x_test,y_train,y_test,maxim,clases

    def entrenar(self):

        x_train, x_test, y_train, y_test, maxim, clases=self.procesar()


        # creacion del modelo de keras, utilizaremos la API funcional para mayor control de los embeddings
        rms=RMSprop(lr=.0013)
        onplateau=ReduceLROnPlateau(patience=1,monitor='val_loss',factor=.5)
        print(" creando modelo. . . ")
        entrada=Input(x_train.shape[1:])
        embb=Embedding(maxim,150)(entrada)
        lstm=Bidirectional(LSTM(80,recurrent_dropout=.2,dropout=.2,return_sequences=True))
        lstm_out=lstm(embb)
        lstm2=LSTM(100,recurrent_dropout=.2,dropout=.2)(lstm_out)
        if len(clases)>2:
            out=Dense(len(clases),activation="softmax")
        else:
            out=Dense(len(clases),activation="sigmoid")
        salida=out(lstm2)
        model=Model(entrada,salida)
        model.compile(loss="sparse_categorical_crossentropy",optimizer=rms,metrics=['acc'])
        for i in range(7):

            history=model.fit(x_train,y_train,epochs=1,batch_size=64,callbacks=[onplateau,],validation_data=[x_test,y_test])
            print(history.history)

        model.save_weights(self.ruta)

        self.model=model
        self.accuracy=float(history.history['val_acc'][0])
        self.save()


    def __str__(self):
        return self.name

