# Generated by Django 4.0.4 on 2022-05-05 06:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0026_alprproto'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alprproto',
            name='mode',
            field=models.CharField(max_length=3),
        ),
    ]