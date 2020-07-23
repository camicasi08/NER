from nltk.tokenize import word_tokenize
import nltk
from nltk import sent_tokenize
import tensorflow as tf
from keras.models import load_model
from keras_contrib.layers.crf import CRF, crf_loss, crf_viterbi_accuracy
from keras.models import model_from_json
import json

from preprocessing import get_vocab, index_sents, SeparateInput, GetCharSentence, tag_input, bio_classification_report
from keras.preprocessing import sequence
import numpy as np
from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger('./corpus/stanford-tagger-4.0.0/stanford-tagger-4.0.0/models/spanish-ud.tagger')




class NERClassifier(object):
    def __init__(self, model_dir, sentence_max_length, max_char_length, corpus, use_pos=False, use_char=True):
        self.model_dir = model_dir
        self.sentence_max_length =  sentence_max_length
        self.max_char_length = max_char_length
        self.corpus = corpus
        self.model = self.load_model()

    def load_model(self):
        print("loading model...")
        if self.corpus == 'conll':
            self.word2idx = np.load('encoded/conll/word2idx.npy',allow_pickle=True).item()
            self.idx2word = np.load('encoded/conll/idx2word.npy',allow_pickle=True).item()
            self.pos2idx = np.load('encoded/conll/pos2idx.npy',allow_pickle=True).item()
            self.idx2pos = np.load('encoded/conll/idx2pos.npy',allow_pickle=True).item()
            self.ner2idx = np.load('encoded/conll/ner2idx.npy',allow_pickle=True).item()
            self.idx2ner = np.load('encoded/conll/idx2ner.npy',allow_pickle=True).item()
            self.char2idx = np.load('encoded/conll/char2idx.npy',allow_pickle=True).item()
            self.idx2char = np.load('encoded/conll/idx2char.npy',allow_pickle=True).item()
            model = load_model(self.model_dir, custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})
        elif self.corpus == 'ancora':
            self.word2idx = np.load('encoded/ancora/word2idx.npy',allow_pickle=True).item()
            self.idx2word = np.load('encoded/ancora/idx2word.npy',allow_pickle=True).item()
            self.pos2idx = np.load('encoded/ancora/pos2idx.npy',allow_pickle=True).item()
            self.idx2pos = np.load('encoded/ancora/idx2pos.npy',allow_pickle=True).item()
            self.ner2idx = np.load('encoded/ancora/ner2idx.npy',allow_pickle=True).item()
            self.idx2ner = np.load('encoded/ancora/idx2ner.npy',allow_pickle=True).item()
            self.char2idx = np.load('encoded/ancora/char2idx.npy',allow_pickle=True).item()
            self.idx2char = np.load('encoded/ancora/idx2char.npy',allow_pickle=True).item()
            with open('./models/model_in_json_ancora.json','r') as f:
                model_json = json.load(f)
            model = model_from_json(model_json,custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})
            model.load_weights(self.model_dir)


        return model

    #30 max-length
    #10 char len
    def preprocess(self, text: str):
        print("preprocessing....")
        sample_sentences = sent_tokenize(text)
        word_sample = []
        pos_sample = []
        if self.corpus == 'conll':
            for input in sample_sentences:
                tmp_words = []
                
                for word in input.split():
                    tmp_words.append(word)
                    
                word_sample.append(tmp_words)
        elif self.corpus == 'ancora':
            postag = [st.tag(s.split()) for s in sample_sentences ]
            for input in postag:
                tmp_words = []
                tmp_pos = []
                for word, pos in input:
                    tmp_words.append(word)
                    tmp_pos.append(pos)
                word_sample.append(tmp_words)
                pos_sample.append(tmp_pos)

        char_info = GetCharSentence(word_sample, self.sentence_max_length, self.max_char_length, self.char2idx)
        sentences_sample_idx = index_sents(word_sample, self.word2idx)

        post_sampple_idx = index_sents(pos_sample, self.pos2idx)
        X_sample_sents = sequence.pad_sequences(sentences_sample_idx, maxlen=self.sentence_max_length, truncating='post', padding='post')
        post_sampple_idx = sequence.pad_sequences(post_sampple_idx, maxlen=self.sentence_max_length, truncating='post', padding='post')

        return word_sample,X_sample_sents, post_sampple_idx, char_info
        
    def predict(self, text: str):
        print("predicting...")
        sample_sentences, X_sample_sents, X_pos_info,  X_char_info = self.preprocess(text)

        if self.corpus == 'conll':
            predictions = self.model.predict([X_sample_sents,np.asarray(X_char_info)])
        elif self.corpus == 'ancora':
            predictions = self.model.predict([X_sample_sents,X_pos_info,np.asarray(X_char_info)])

        predictions = np.argmax(predictions, axis=-1)

        result = [[self.idx2ner[t] for t in s] for s in predictions]
        results ={}
        for idx, sent in enumerate(sample_sentences):
            #print("idx", str(idx))
            results[idx] = []
            for j, tag in enumerate(result[idx]):
            #print("J", str(j))
                try:
                    results[idx].append({"Word": sample_sentences[idx][j], "Tag":tag})
                except:
                    continue
                #results[idx].append({'Word': 'PAD', "Tag": 'O'})

        return results


#NER = NERClassifier('./models/10_0.25_400_3_0.001_64_conllfff.h5', 30, 10)
#print(NER.predict("HOLA MUNDO"))