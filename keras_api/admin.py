# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from keras_api.models import keras_model,NLP_sets

# Register your models here.

class keras_model_admin(admin.ModelAdmin):
    list_display = ('id','name','accuracy','ruta')
    search_fields = ['name']
    actions = ['entrenar_keras','cargar_modelo' ]
    def entrenar_keras(self,request,queryset):
        for modelo in queryset:
            model=modelo
            model.entrenar()
    def cargar_modelo(self,request,queryset):
        for modelo in queryset:
            model=modelo
            modelo.model.load_weights(modelo.ruta)
class nlp_sets_admin(admin.ModelAdmin):
    list_display = ('id','name','category','Q_targets')
    search_fields = ['name']





admin.site.register(keras_model,keras_model_admin)
admin.site.register(NLP_sets,nlp_sets_admin)