# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:41:22 2018

@author: hasee
"""
'''
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
jar = 'stanford-ner.jar'
model = 'english.conll.4class.distsim.crf.ser.gz'
st = StanfordNERTagger(model,jar)

def entity_recognize(sentence):
    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    return classified_text

print entity_recognize("Charlie is in dasd the a China")

print word_tokenize("Charlie is in dasd the a China")
'''
query = "Charlie is in dasd the a China"
print query.index("is in")

dict = {'Name': 'Runoob', 'Age': 27}

print ("Age 值为 : %s" %  dict.get('Age'))
print ("Sex 值为 : %s" %  dict.get('Sex', "NA"))
'''
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz') 
print st.tag('Rami is  studying at Stony Brook University in China'.split()) 
'''