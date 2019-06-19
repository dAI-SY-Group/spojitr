#! /usr/bin/env sh

###########################################################
# Data

## NLTK
python3 << EOPYTHON
import nltk

nltk.download("stopwords")
nltk.download("word_tokenize")
nltk.download("punkt")
EOPYTHON

###########################################################
# 3rd party

## weka
cd /data/tmp
curl https://netix.dl.sourceforge.net/project/weka/weka-3-8/3.8.3/weka-3-8-3.zip -O
unzip weka-3-8-3.zip
cp weka-3-8-3/weka.jar /data/spojitr_install/3rd/

## spojit
cd /data/tmp
git clone https://github.com/michaelrath-work/spojit.git
cd spojit
python3 setup.py bdist_wheel
cp -v dist/spojit-*.whl /data/spojitr_install/3rd
pip3 install /data/spojitr_install/3rd/spojit-*.whl

## weka run helper
cd /data/tmp
git clone https://github.com/michaelrath-work/weka-run-helper.git
cp -v weka-run-helper/run_weka.py /data/spojitr_install/3rd


###########################################################
# Setup git

echo ".spojitr/" >> /root/.gitignore_global
git config --global user.email "user@spojitr.com"
git config --global user.name "Spojitr User"
git config --global core.excludesfile /root/.gitignore_global

