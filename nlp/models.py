import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Convolution1D, GlobalMaxPool1D, Dropout
from keras import optimizers
from keras.models import load_model

import json, argparse, os
import re
import io
import sys

class all_models():
    def __init__(self, embeddingMatrix=None, embedding_dim=None, lstm_dim=None, 
                 dropout=None, n_class=None, max_seq_length=None, lr=None):
        self.embeddingMatrix = embeddingMatrix
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.n_class = n_class
        self.max_seq_length = max_seq_length
        self.lr = lr

    def __call__(self, embeddingMatrix, model_name="lstm_simple"):
        if model_name=="lstm_simple":
            return self.lstm_simple(embeddingMatrix)

    def lstm_simple(self, embeddingMatrix):
        """Constructs the architecture of the model
        Input:
            embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
        Output:
            model : A basic LSTM model
        """
        embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                    self.embedding_dim,
                                    weights=[embeddingMatrix],
                                    input_length=self.max_seq_length,
                                    trainable=False)
        model = Sequential()
        model.add(embeddingLayer)
        model.add(LSTM(self.lstm_dim, dropout=self.dropout))
        model.add(Dense(self.n_class, activation='sigmoid'))
        
        rmsprop = optimizers.rmsprop(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                    optimizer=rmsprop,
                    metrics=['acc'])
        print(model.summary())
        return model




