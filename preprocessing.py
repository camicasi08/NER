import codecs, re, random
from collections import Counter
import numpy as np
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

# function to get vocab, maxvocab
# takes sents : list (tokenized lists of sentences)
# takes maxvocab : int (maximum vocab size incl. UNK, PAD
# takes stoplist : list (words to ignore)
# returns vocab_dict (word to index), inv_vocab_dict (index to word)
def get_vocab(sent_toks, maxvocab=10000, min_count=1, stoplist=[], unk='UNK', pad='PAD', verbose=False):
    # get vocab list
    vocab = [word for sent in sent_toks for word in sent]
    sorted_vocab = sorted(Counter(vocab).most_common(), key=lambda x: x[1], reverse=True)
    sorted_vocab = [i for i in sorted_vocab if i[0] not in stoplist and i[0] != unk]
    if verbose:
        print("total vocab:", len(sorted_vocab))
    sorted_vocab = [i for i in sorted_vocab if i[1] >= min_count]
    if verbose:
        print("vocab over min_count:", len(sorted_vocab))
    # reserve for PAD and UNK
    sorted_vocab = [i[0] for i in sorted_vocab[:maxvocab - 2]]
    vocab_dict = {k: v + 1 for v, k in enumerate(sorted_vocab)}
    vocab_dict[unk] = len(sorted_vocab) + 1
    vocab_dict[pad] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

    return vocab_dict, inv_vocab_dict


# function to convert sents to indexed vectors
# takes list : sents (tokenized sentences)
# takes dict : vocab (word to idx mapping)
# returns list of lists of indexed sentences
def index_sents(sent_tokens, vocab_dict, reverse=False, unk_name='UNK', verbose=False):
    vectors = []
    for sent in sent_tokens:
        sent_vect = []
        if reverse:
            sent = sent[::-1]
        for word in sent:
            if word in vocab_dict.keys():
                sent_vect.append(vocab_dict[word])
            else:  # out of max_vocab range or OOV
                sent_vect.append(vocab_dict[unk_name])
        vectors.append(np.asarray(sent_vect))
    vectors = np.asarray(vectors)
    return vectors


# decode an integer-indexed sequence
# takes indexed_list : one integer-indexedf sentence (list or array)
# takes inv_vocab_dict : dict (index to word)
# returns list of string tokens
def decode_sequence(indexed_list, inv_vocab_dict):
    str = []
    for idx in indexed_list:
        # print(intr)
        str.append(inv_vocab_dict[int(idx)])
    return(str)

###[(word, pos)]
def tag_input(model, sentences, vocab_word,vocab_pos,idx2ner, max_len):
  word_sample = []
  pos_sample = []
  for input in sentences:
    tmp_words = []
    tmp_pos = []
    for word, pos in input:
      tmp_words.append(word)
      tmp_pos.append(pos)
    word_sample.append(tmp_words)
    pos_sample.append(tmp_pos)

  ##cración de indices
  sentences_sample_idx = index_sents(word_sample, vocab_word)
  post_sampple_idx = index_sents(pos_sample, vocab_pos)

  ## Padding
  X_sample_sents = sequence.pad_sequences(sentences_sample_idx, maxlen=max_len, truncating='post', padding='post')
  post_sampple_idx = sequence.pad_sequences(post_sampple_idx, maxlen=max_len, truncating='post', padding='post')

  ## predicción
  predictions = model.predict([X_sample_sents,post_sampple_idx])
  predictions = np.argmax(predictions, axis=-1)

  result = [[idx2ner[t] for t in s] for s in predictions]
  results ={}
  for idx, sent in enumerate(word_sample):
    #print("idx", str(idx))
    results[idx] = []
    for j, tag in enumerate(result[idx]):
      #print("J", str(j))
      try:
        results[idx].append({"Word": word_sample[idx][j], "Tag":tag})
      except:
        continue
        #results[idx].append({'Word': 'PAD', "Tag": 'O'})
    



  return results



def SeparateInput(dataInput):
    '''
        SeparateInput()
        dataInput: DataFrame(Sentence #, Word, POS, Tag)
        return sentences_text, sentences_pos, sentences_ner
    '''
    sentmarks = dataInput["Sentence #"].unique().tolist()
    #print(len(sentmarks))

    sentence_text = dataInput.groupby("Sentence #")['Word'].apply(list)
    sentence_post = dataInput.groupby("Sentence #")['POS'].apply(list)
    sentence_ners = dataInput.groupby("Sentence #")['Tag'].apply(list)
    #print(sentmarks[:10])
    '''
    for idx, s in enumerate(sentmarks):
        sentence_text.append(dataInput[dataInput['Sentence #'] == s]['Word'].tolist())
        sentence_post.append(dataInput[dataInput['Sentence #'] == s]['POS'].tolist())
        sentence_ners.append(dataInput[dataInput['Sentence #'] == s]['Tag'].tolist())'''
    return sentence_text, sentence_post, sentence_ners


def GetCharSentence(sentences, MAX_LENGTH, max_len_char, char2Idx):
    '''
        Tokens to chars by sentence
    '''
    char_sentences = []
    for sent in sentences:
        char_sent = []
        for i in range(MAX_LENGTH):
            char_token = []
            for j in range(max_len_char):
            #print(j)
                try:
                    char = sent[i][j]

                    char_token.append(char2Idx.get(char))
                except:
                    char_token.append(char2Idx.get('PAD'))
                #print('PAD')
            char_sent.append(char_token)
        char_sentences.append(np.array(char_sent))
    return char_sentences




def bio_classification_report(y_true, y_pred):
    """
    from scrapinghub's python-crfsuite example
    
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O', 'PAD'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )