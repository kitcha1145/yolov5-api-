# Generated by Django 4.0.4 on 2022-04-18 02:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0002_student_tracks'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student',
            name='tracks',
        ),
    ]
