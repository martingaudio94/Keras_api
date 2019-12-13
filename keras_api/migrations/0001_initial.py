# Generated by Django 2.1.5 on 2019-12-13 14:47

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='keras_model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=253)),
                ('accuracy', models.FloatField(default=0)),
                ('active', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='NLP_sets',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=253)),
                ('Q_targets', models.IntegerField(default=0)),
                ('text', models.CharField(default='', max_length=65535)),
                ('category', models.CharField(default='', max_length=253)),
            ],
        ),
    ]
