# Generated by Django 4.0.4 on 2022-04-19 02:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0021_imageinferproto_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageinferproto',
            name='name',
            field=models.CharField(max_length=100),
        ),
    ]