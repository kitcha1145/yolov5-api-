# Generated by Django 4.0.4 on 2022-04-19 02:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0018_alter_imageinferproto_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageinferproto',
            name='image',
            field=models.BinaryField(),
        ),
    ]
