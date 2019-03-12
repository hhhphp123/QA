# -*- coding: utf-8 -*-
#coding=utf-8
"""
Created on Tue May 22 13:39:20 2018

@author: hasee
"""
#firsr of all import all package and load the file
#coding=utf-8
import spacy
from collections import OrderedDict
from operator import itemgetter 
import string
from math import log
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter
from nltk.corpus import stopwords    
from nltk.tag import StanfordNERTagger
import json
import nltk
import csv
#nltk.download('averaged_perceptron_tagger')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
jar = 'stanford-ner.jar'
model = 'english.conll.4class.distsim.crf.ser.gz'
model1 = 'english.all.3class.distsim.crf.ser.gz'
model2 = 'english.muc.7class.distsim.crf.ser.gz'
punc = string.punctuation
stopwordsPart = set(stopwords.words('english'))
stopwordsPart.remove('the')  
stopwordsPart.remove('of') 
stopwordsAll = set(stopwords.words('english'))

def opne_json(text):
    with open(text,'r') as input_file:
        document = json.load(input_file)
    return document



def get_tag_model(model,jar):
    return StanfordNERTagger(model,jar)

person_model = get_tag_model(model,jar)
person_model2 = get_tag_model(model1,jar)
number_model = get_tag_model(model2,jar)




documents_dict = opne_json("documents.json")
test_dict = opne_json("testing.json")
dev_dict = opne_json("devel.json")
train_dict = opne_json("training.json")
query_lables = opne_json("QuestionLabel.json")

#get paragraph from the training data
def get_paragraph(docid,documents_dict):
    #get the paragraph that contains the answer
    for i in documents_dict:
        if i['docid'] == docid:
            document = i['text']
            break
    return document


#get TF 
def term_freqs(document):
    tfs = defaultdict(dict)
    tfs_forward = defaultdict(dict)
    doc_id = 0
    for sentence in document:
        for token in word_tokenize(sentence):
            if token not in stopwordsAll and token not in punc:  
                term = lemmatizer.lemmatize(token.lower())
                tfs[term][doc_id] = tfs[term].get(doc_id, 0) + 1 
                tfs_forward[doc_id][term] = tfs[doc_id].get(term, 0) + 1 
        doc_id += 1
    return tfs,doc_id+1,tfs_forward

#build TF_IDF model
def get_tfidf(tfs, total_docment,tfs_forward):
    document_length = {}
    for doc_id,doc_list in tfs_forward.items():
        length = 0
        for term, freq in doc_list.items():
            length += freq ** 2
        length = length **0.5
        document_length[doc_id] =  length
    tfidf = defaultdict(dict)
    for term, doc_list in tfs.items():
        df = len(doc_list)
        for doc_id, freq in doc_list.items(): 
            tfidf[term][doc_id] = (float(tfs[term][doc_id]) * log(total_docment / df))# / document_length[doc_id]
    return tfidf

def get_okapibm25(tf, total_docment, documents):
    '''Calculate and return term weights based on okapibm25'''
    k1, b, k3 = 1.5, 0.5, 0
    okapibm25 = defaultdict(dict)

    # calculate average doc length 
    total = 0
    for d in documents:
        total += len(d)
    avg_doc_length = total/len(documents)*1.0

    for term, doc_list in tf.items():
        df = len(doc_list)
        for doc_id, freq in doc_list.items():
            # term occurences in query
            # qtf = question.count(term) # SEPCIAL 
            qtf = 1.2
            idf = log((total_docment-df+0.5) / df+0.5)
            tf_Dt = ((k1+1)*tf[term][doc_id]) / (k1*((1-b)+b*(len(documents[doc_id])/avg_doc_length) + tf[term][doc_id]))
            if qtf == 0:
                third = 0
            else:
                third = ((k3+1)*qtf) / (k3+qtf)
                okapibm25[term][doc_id] = idf*tf_Dt*third

    return okapibm25

#find top_k paragraph that may contain the answer
def get_top_k_document(tfidf,query,k,document):
    top_document_id = Counter()
    for token in word_tokenize(query):
        if token not in stopwordsAll:  
             term = lemmatizer.lemmatize(token.lower())
             term_tfidf = tfidf[term]
             for docid, weight in term_tfidf.items():
                 top_document_id[docid] += weight
    top_document_id = top_document_id.most_common(k)
    top_document = []
    for document_id,weight in top_document_id:
        top_document.append(document[document_id])
    return top_document
#filter the key words in query
#filter the key words in query
def get_open_class_word(query):
    #query = nltk.word_tokenize(query)
    #tagged = nltk.pos_tag(query)#nltk.word_tokenize(query))#, tagset="universal")
    tagged = nltk.pos_tag(word_tokenize(query), tagset="universal")
    return [p[0] for p in tagged if p[1] in ["NOUN","VERB","NUM"] and p[0] not in stopwordsAll]
    #return [p[0] for p in tagged if p[1] in ["NN","NNP","NNS","NP","VB","VBD","CD","JJ",] and p[0] not in stopwordsAll]

#combine the NER with same tag
def same_tag(ner_output):
    word,tag = 'the','O'
    combo = []
    for word1,tag1 in ner_output:
        '''
        if tag1 == "O" and word1 not in stopwordsAll and word1 not in punc:
            combo.append((word,tag))
            tag = tag1
            word = word1
            continue
            '''
        if tag1 == tag:
            if word[-1] in ['(',')']:
                word += word1
            if word1 in [')']:
                 word += word1
            else:     
                word += " " + word1
        else:
            combo.append((word,tag))
            tag = tag1
            word = word1
            continue
    if len(combo) != 0:
        combo.pop(0)
    return combo


def same_tag_other(ner_output):
    word,tag = 'the','O'
    combo = []
    for word1,tag1 in ner_output:
        '''
        if tag1 == "O" and word1 not in stopwordsAll and word1 not in punc:
            combo.append((word,tag))
            tag = tag1
            word = word1
            continue
            '''
        if tag1 in ["NOUN","ADJ","NUM"] and tag in ["NOUN"]:
        #if tag in ["NN","NNP","JJ","CD","CC","NNS","NP","IN"] and tag1 in ["NN","NNP","JJ","CD","CC","NNS","NP","IN"]:
        #if tag in ["NN","NNP","JJ","CD","NNS","NNPS"] and tag1 in ["NN","NNP","NNS","NNPS"]:
            if word[-1] in ['(',')']:
                word += word1
            if word1 in [')']:
                 word += word1
            else:     
                word += " " + word1
            tag = tag1
        else:
            combo.append((word,tag))
            tag = tag1
            word = word1
            continue
    if len(combo) != 0:
        combo.pop(0)
    return combo

def most_in(key_words,sentence):
    all_in = True
    len1 = len(key_words)
    word_in = 0
    for i in key_words:
        try:
            index = sentence.index(i)
            word_in += 1 
        except ValueError:
            continue
    return len1 < 2*word_in

def in_key_words(word,key_words):
    in_key = False
    for i in key_words:
        if word.find(i) != -1:       
            in_key = True
            break
    return in_key
            
rules = {
            'which': 'PERSON',
            'name':'PERSON',
            'country': 'LOCATION',
            'capital': 'LOCATION',
            'newspaper':'ORGANIZATION',
            'company':'ORGANIZATION',
            'city': 'LOCATION',
            'person':'PERSON',
            'location': 'LOCATION',
            'mountain':'LOCATION',
            'website':'ORGANIZATION',
            'airline':'ORGANIZATION',
            'which organization': 'ORGANIZATION',
            'where': 'LOCATION',
            'when': 'DATE',           
            'who': 'PERSON',     
            'what scientist':'PERSON',
            'what time':'TIME',
            'what athlete':'PERSON',
            'which athlete':'PERSON',
            'what people': 'PERSON',
            'what date':'DATE',
            'what day':'DATE',
            'what year': 'DATE',
            'what city' : 'LOCATION',
            'which company': 'ORGANIZATION' ,
            'which publication':'ORGANIZATION',
            'what government':'ORGANIZATION',
            'which supporters' : 'PERSON',
            'which footballer': 'PERSON',
            'which actor':'PERSON',
            'Which actress':'PERSON',
            'which American actress':'PERSON',
            'what activists':'PERSON',
            'which team member' : 'PERSON',
            'what football star': 'PERSON',
            'which blogger': 'PERSON',
            'which torchbearer':'PERSON',
            'which wheelchair-bound torchbearer' : 'PERSON',
            'how much of': 'PERCENT',
            'by how much': 'PERCENT',
            'how much': 'MONEY'
            
            
}

money_list = ['cost', 'worth', 'spend', 'money', 'worth', 'invest']

def tag_answer_type(question):
    answer_type = 'O'
    processed_question = []
    processed_question_str = None
    for token in [question]:
        processed_question.append(token.lower())
    processed_question_str = " ".join(x for x in processed_question)
    for k,v in rules.items():
        if k in processed_question_str:
            #print(k)
            if k == 'how much':
                for item in money_list:
                    #print("item", item)
                    if item in processed_question_str:
                        answer_type = 'MONEY'
                    else:
                        continue
            else:
                answer_type = rules.get(k, "O")    
    return answer_type

def get_answer_list(query,top_k):
    key_words = get_open_class_word(query)
    answer_list = {}
    answer_type = tag_answer_type(query)
    for ans_sentence in top_k:

        #if most_in(key_words,ans_sentence) == False:
          #  continue
        if answer_type == "O":
           
            #word_list = nltk.word_tokenize(ans_sentence)
            #word_list_tag = nltk.pos_tag(word_tokenize(ans_sentence), tagset="universal")
            #word_list_tag = same_tag_other(word_list_tag)
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(ans_sentence)
            word_list = doc.noun_chunks
            word_list_tag = []
            for i in word_list:
                word_list_tag.append((i.text,"NOUN"))
            answer_type = ["NOUN","NUM"] 
            #word_list_tag = nltk.pos_tag(word_list)
            #word_list_tag = same_tag_other(word_list_tag)
            #answer_type = ["NN","NNP","NNS","NNPS","CD"]
            '''
            if "how" in query:
                answer_type = ["CD"]
            else:
                answer_type = ["NN","NNP","NNS","NP"]
            '''    
        else:
            word_list =  []
            for word in word_tokenize(ans_sentence):
                word_list.append(word)    
            word_list_tag = number_model.tag(word_list)
            word_list_tag = same_tag(word_list_tag)
            answer_type = [answer_type]
        for word,tags in word_list_tag:
            if word not in answer_list.keys():
                if word not in stopwordsAll and word not in punc and tags in answer_type and word not in key_words and in_key_words(word,key_words) == False:
                    distance_list = []
                    distance = 0
                    for key_word in key_words:
                        try:
                            index = ans_sentence.index(key_word)
                            distance_list.append(index)
                        except ValueError:
                            distance_list.append(5000)
                    for index in distance_list:
                        try:
                            distance += abs(index - ans_sentence.index(word))
                        except ValueError:
                            continue
                    answer_list[word] = distance
    if  len(answer_list.items()) != 0:
        #return sorted(answer_list.items(), lambda x, y: cmp(x[1], y[1]))[0][0].lower()
        return sorted(answer_list.items(), key = itemgetter(1), reverse = False)[0][0].lower()
    else:
        return None
    
    


        
def output(test_dict,documents_dict):
    output = open("out.txt", "w")
    docidstart = train_dict[0]['docid']
    document = get_paragraph(docidstart,documents_dict)
    tfs,total_docment,tfs_forward = term_freqs(document)
    tfidf = get_tfidf(tfs, total_docment,document)
    o = 0
    for i in test_dict:
        docid = i['docid']
        if docid != docidstart:
            document = get_paragraph(docid,documents_dict)
            tfs,total_docment,tfs_forward = term_freqs(document)
            tfidf = get_tfidf(tfs, total_docment,document)
            docidstart = docid
        query = i['question']
        qaid = i['id']
        top_k = get_top_k_document(tfidf,query,1,document)
        potiential_answer = get_answer_list(query,top_k)
        try:
            output.write(str(qaid) + "," +str(potiential_answer) + '\n')
            print (str(qaid))
        except UnicodeEncodeError:
            output.write(str(qaid) + '\n')
            print (potiential_answer)
            
def output_csv(test_dict,documents_dict):
    '''
    csv_file = open('outpur.csv', 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'answer'])
    '''

    output = open("out_new.txt", "w")
    docidstart = train_dict[0]['docid']
    document = get_paragraph(docidstart,documents_dict)
    tfs,total_docment,tfs_forward = term_freqs(document)
    #tfidf = get_okapibm25(tfs, total_docment,tfs_forward)
    tfidf = get_okapibm25(tfs, total_docment,document) # 25 model
    for i in test_dict[0:]:
        docid = i['docid']
        if docid != docidstart:
            document = get_paragraph(docid,documents_dict)
            tfs,total_docment,tfs_forward = term_freqs(document)
            #tfidf = get_okapibm25(tfs, total_docment,tfs_forward)
            tfidf = get_okapibm25(tfs, total_docment,document)
            docidstart = docid
        query = i['question']
        qaid = i['id']
        top_k = get_top_k_document(tfidf,query,1,document)
        potiential_answer = get_answer_list(query,top_k)
        #potiential_answer = potiential_answer.encode('ascii', 'ignore').decode('ascii')
        try:#.encode('utf-8')
            output.write(str(potiential_answer) + "\n")
            print (str(qaid))
        except UnicodeEncodeError:
            output.write(str(qaid) + '\n')
            print (potiential_answer)
            print (str(qaid))
     
        
        
       
def get_doc_accuracy(train_dict,documents_dict):
    total = 0
    right = 0
    docidstart = train_dict[0]['docid']
    document = get_paragraph(docidstart,documents_dict)
    tfs,total_docment,tfs_forward = term_freqs(document)
    tfidf = get_okapibm25(tfs, total_docment,document)
    for i in train_dict:
        docid = i['docid']
        if docid != docidstart:
            print (docid)
            document = get_paragraph(docid,documents_dict)
            tfs,total_docment,tfs_forward = term_freqs(document)
            tfidf = get_okapibm25(tfs, total_docment,document)
            docidstart = docid
        query = i['question']
        answer_paragraph = i['answer_paragraph']
        answer_sentence = documents_dict[docid]['text'][answer_paragraph]
        top_k = get_top_k_document(tfidf,query,2,document)
        if answer_sentence in top_k:
            right += 1
        total += 1
    print (right)
    print (total)
    return (float(right)/total)

      
def get_accuracy(train_dict,documents_dict):
    docidstart = train_dict[0]['docid']
    document = get_paragraph(docidstart,documents_dict)
    tfs,total_docment,tfs_forward = term_freqs(document)
    tfidf = get_tfidf(tfs, total_docment,tfs_forward)
    right,total = 0,0
    for i in train_dict[0:]:
        docid = i['docid']
        if docid != docidstart:
            document = get_paragraph(docid,documents_dict)
            tfs,total_docment,tfs_forward = term_freqs(document)
            tfidf = get_tfidf(tfs, total_docment,tfs_forward)
            docidstart = docid
        query = i['question']
        #qaid = i['id']
        answer = i['text']
        top_k = get_top_k_document(tfidf,query,3,document)
        potiential_answer = get_answer_list(query,top_k)
        print (potiential_answer)
        print (answer)
        if answer == potiential_answer:
            right += 1
        total += 1
        if total == 50:
            print (total)
            print (right)
            break

#output_csv(test_dict,documents_dict)
output_csv(test_dict,documents_dict)