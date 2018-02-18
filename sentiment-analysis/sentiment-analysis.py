#!/usr/bin/python3
# -*- coding ; utf-8 -*-

'''
    # Classificação de sentimento
    
    O objetivo do algoritmo é identificar um sentimento positivo ou negativo
    data uma frase qualquer.

    ## Base de Dados
    
    A base de dados é composta de dois arquivos, um contendo **frases positivas"" e
    outro contendo **frases negativa**. 

    ## Ideia
    
    Não se pode trabalhar diretamente com texto no tensorflow, é necessário que
    o texto seja convertido para alguma sequência numérica em vez de sequência de
    caracteres.

    Será contruído um dicionário léxico, as principais pelavras da bases de dados
    serão selecionadas e utilizadas para converter as frases em sequências numéricas.
    Por exemplo:

    **dicionário léxico** -> [gato, rato, cavalo, pulo]
    **frase** -> "O *gato* comeu o *rato*"
    **sequência numérica** -> [1,1,0,0] 

    *Ver o arquivo [preprocessing.py](preprocessing.py)
'''

import tensorflow as tf
import numpy as np
from preprocessing import create_features_sets_and_labels

train_x, train_y, test_x, test_y = create_features_sets_and_labels('database/pos.txt', 'database/neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

# height x width
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

'''
    ## Modelo da Rede Neural

    O modelo da rede neural é o mesmo do exemplo [Simple Neural Network](../simpleNN)
'''

def neural_network_model(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hl2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hl3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    #(input_data * weights) + biases
    
    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    return output

def train_nn(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    n_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #train
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            i = 0

            while i < len(train_x):
                start = i
                end = i + batch_size

                epoch_x = np.array(train_x[start:end])
                epoch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost],
                                feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch ', epoch, 'of ', n_epochs, '\nloss: ', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: test_x,
                                           y: test_y}))

train_nn(x, y)