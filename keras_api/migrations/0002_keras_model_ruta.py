# Generated by Django 2.1.5 on 2019-12-16 15:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('keras_api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='keras_model',
            name='ruta',
            field=models.CharField(blank=True, max_length=253),
        ),
    ]