# Generated by Django 4.0.4 on 2022-04-18 09:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiApp', '0015_usermanagement_credit_limit'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageInferProto',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.CharField(max_length=10000)),
            ],
        ),
    ]
