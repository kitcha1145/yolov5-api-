# Generated by Django 4.0.4 on 2022-04-19 03:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0022_alter_imageinferproto_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='usermanagement',
            name='age',
        ),
    ]
