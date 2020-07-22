from nltk.tokenize import word_tokenize
import nltk
from nltk import sent_tokenize
import tensorflow as tf
from keras.models import load_model
from keras_contrib.layers.crf import CRF, crf_loss, crf_viterbi_accuracy

from preprocessing import get_vocab, index_sents, SeparateInput, GetCharSentence, tag_input, bio_classification_report
from keras.preprocessing import sequence
import numpy as np


class NERClassifier(object):
    def __init__(self, model_dir, sentence_max_length, max_char_length):
        self.model_dir = model_dir
        self.sentence_max_length =  sentence_max_length
        self.max_char_length = max_char_length
        self.model = self.load_model()

    def load_model(self):
        print("loading model...")
        self.word2idx = np.load('encoded/word2idx.npy',allow_pickle=True).item()
        self.idx2word = np.load('encoded/idx2word.npy',allow_pickle=True).item()
        self.pos2idx = np.load('encoded/pos2idx.npy',allow_pickle=True).item()
        self.idx2pos = np.load('encoded/idx2pos.npy',allow_pickle=True).item()
        self.ner2idx = np.load('encoded/ner2idx.npy',allow_pickle=True).item()
        self.idx2ner = np.load('encoded/idx2ner.npy',allow_pickle=True).item()
        self.char2idx = np.load('encoded/char2idx.npy',allow_pickle=True).item()
        self.idx2char = np.load('encoded/idx2char.npy',allow_pickle=True).item()
        model = load_model(self.model_dir, custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})

        return model

    #30 max-length
    #10 char len
    def preprocess(self, text: str):
        print("preprocessing....")
        sample_sentences = sent_tokenize(text)
        word_sample = []
        for input in sample_sentences:
            tmp_words = []
            
            for word in input.split():
                tmp_words.append(word)
                
            word_sample.append(tmp_words)
           
        char_info = GetCharSentence(word_sample, self.sentence_max_length, self.max_char_length, self.char2idx)
        sentences_sample_idx = index_sents(word_sample, self.word2idx)
        X_sample_sents = sequence.pad_sequences(sentences_sample_idx, maxlen=self.sentence_max_length, truncating='post', padding='post')

        return word_sample,X_sample_sents, char_info
        
    def predict(self, text: str):
        print("predicting...")
        sample_sentences, X_sample_sents, char_info = self.preprocess(text)

        predictions = self.model.predict([X_sample_sents,np.asarray(char_info)])
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