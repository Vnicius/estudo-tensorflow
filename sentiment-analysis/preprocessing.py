#!/usr/bin/python3
# -*- coding : utf-8 -*-

'''
    # Pré-processamento
    
    Não se pode trabalhar diretamente com texto no tensorflow, é necessário que
    o texto seja convertido para alguma sequência numérica em vez de sequência de
    caracteres.

    Será contruído um dicionário léxico, as principais pelavras da bases de dados
    serão selecionadas e utilizadas para converter as frases em sequências numéricas.
    Por exemplo:

    **dicionário léxico** -> [gato, rato, cavalo, pulo]
    **frase** -> "O *gato* comeu o *rato*"
    **sequência numérica** -> [1,1,0,0] 
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
n_lines = 2000

def create_lexicon(pos, neg):
    lexicon = []
    
    for fl in [pos, neg]:
        with open(fl, 'r') as f:
            it = (line.replace('\n', '').lower() for line in f)
            i = 0
            while i < n_lines:
                # separar todas as palavras da sentença
                all_words = word_tokenize(next(it))
                # insere todas as palavras na lista
                lexicon += list(all_words)
                i += 1
    
    # converter as palavras para a forma radical
    lexicon = [lemmatizer.lemmatize(word) for word in lexicon]
    word_counts = Counter(lexicon)
    lexicon = []

    for w in word_counts:
        # ignora as palvras mais comuns
        if 1000 > word_counts[w] and word_counts[w] > 50 and w.isalpha():
            lexicon.append(w)

    return lexicon

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        it = (line.replace('\n', '').lower() for line in f)
        i = 0

        while i < n_lines:
            current_words = word_tokenize(next(it))
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] = 1
            
            features = list(features)
            featureset.append([features, classification])
            i += 1
    
    return featureset

def create_features_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

# print(create_features_sets_and_labels('database/pos.txt', 'database/neg.txt'))