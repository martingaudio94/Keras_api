# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from keras_api.models import keras_model,NLP_sets

# Register your models here.

class keras_model_admin(admin.ModelAdmin):
    list_display = ('id','name')
    search_fields = ['name']

class nlp_sets_admin(admin.ModelAdmin):
    list_display = ('id','name','category','Q_targets')
    search_fields = ['name']


admin.site.register(keras_model,keras_model_admin)
admin.site.register(NLP_sets,nlp_sets_admin)