import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from nlp.preprocessing import preprocessData, writeNormalisedData
from nlp.models import all_models

import json, argparse, os
import re
import io
import sys


class classifier():

    def __init__(self, lr=0.001,
                 n_folds=5, # Change default values of following to none and provide default in application
                 trainDataPath="/home/singh/Desktop/emocontext/starterkitdata/train.txt",
                 testDataPath="/home/singh/Desktop/emocontext/starterkitdata/devwithoutlabels.txt",
                 solutionPath="/home/singh/Desktop/emocontext/starterkitdata/test/test11.txt",
                 gloveDir="/home/singh/Desktop/emocontext/glove.6B/",
                 n_class=4,
                 max_nb_words=20000,
                 max_seq_length=100,
                 embedding_dim=100,
                 batch_size=32,
                 lstm_dim=200,
                 dropout=0.2,
                 n_epochs=10,
                 eval_kfold=True,
                 model = None
                 ):

        self.lr = lr
        self.n_folds = n_folds
        self.trainDataPath = trainDataPath
        self.testDataPath = testDataPath
        self.solutionPath = solutionPath
        self.gloveDir = gloveDir
        self.n_class = n_class
        self.max_nb_words = max_nb_words
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.eval_kfold = eval_kfold
        if model is not None:
            self.model = model
        else:
            self.model = all_models(embedding_dim=self.embedding_dim, 
                                    lstm_dim=self.lstm_dim, 
                                    dropout=self.dropout,
                                    n_class=self.n_class,
                                    max_seq_length=self.max_seq_length,
                                    lr=self.lr)

    def fit(self):
        self.__call__()

    def __call__(self):
        self.evaluate()

    def evaluate(self):
        # Load data from given file data

        print("Processing training data...")
        trainIndices, trainTexts, labels = preprocessData(self.trainDataPath, mode="train")
        # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
        # writeNormalisedData(self.trainDataPath, trainTexts)
        print("Processing test data...")
        testIndices, testTexts = preprocessData(self.testDataPath, mode="test")
        # writeNormalisedData(self.testDataPath, testTexts)

        # Preprocess the data
        print("Extracting tokens...")
        tokenizer = Tokenizer(num_words=self.max_nb_words)
        tokenizer.fit_on_texts(trainTexts)
        trainSequences = tokenizer.texts_to_sequences(trainTexts)
        testSequences = tokenizer.texts_to_sequences(testTexts)

        wordIndex = tokenizer.word_index
        print("Found %s unique tokens." % len(wordIndex))

        print("Populating embedding matrix...")
        embeddingMatrix = getEmbeddingMatrix(wordIndex, self.gloveDir, self.embedding_dim)

        data = pad_sequences(trainSequences, maxlen=self.max_seq_length)
        label1 = np.asarray(labels)
        labels = to_categorical(np.asarray(labels))
        print("Shape of training data tensor: ", data.shape)
        print("Shape of label tensor: ", labels.shape)

        # Randomize data
        np.random.seed(30)
        np.random.shuffle(trainIndices)
        data = data[trainIndices]
        labels = labels[trainIndices]
        label1 = label1[trainIndices]

        # Evaluate kfold
        if self.eval_kfold:
            self.evaluate_kfold(data=data, labels=labels, label1 = label1, embeddingMatrix=embeddingMatrix)

        # Train on entire data
        model = self.train(X_train=data, y_train=labels,
                           embeddingMatrix=embeddingMatrix)

        # Save and load model
        model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
        # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

        # Write predictions to file
        print("Creating solution file...")
        testData = pad_sequences(testSequences, maxlen=self.max_seq_length)
        predictions = model.predict(testData, batch_size=self.batch_size)
        predictions = predictions.argmax(axis=1)

        with io.open(solutionPath, "w", encoding="utf8") as fout:
            fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
            with io.open(testDataPath, encoding="utf8") as fin:
                fin.readline()
                for lineNum, line in enumerate(fin):
                    fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                    fout.write(label2emotion[predictions[lineNum]] + '\n')
        print("Completed. Model parameters: ")
        print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
            % (self.lr, self.lstm_dim, self.dropout, self.batch_size))



    def evaluate_kfold(self, data=None, labels=None, label1=None, embeddingMatrix=None):
        # Add training
        metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=30)
        skf.get_n_splits(data, label1)
    
        print("Starting k-fold cross validation...")
        k=0
        for train_index, test_index in skf.split(data, label1):
            print('-'*40)
            print("Fold %d/%d" % (k+1, self.n_folds))
            k=k+1

            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            print("Building model...")
            model = self.train(X_train=X_train, y_train=y_train,
                               embeddingMatrix=embeddingMatrix,
                               X_test = X_test, y_test = y_test
                               )
            
            predictions = model.predict(xVal, batch_size=BATCH_SIZE)
            accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
            metrics["accuracy"].append(accuracy)
            metrics["microPrecision"].append(microPrecision)
            metrics["microRecall"].append(microRecall)
            metrics["microF1"].append(microF1)
            

        print("\n============= Metrics =================")
        print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
        print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
        print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
        print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

        print("\n======================================")

    def train(self, X_train=None, y_train=None, embeddingMatrix=None, X_test=None, y_test=None):
        model_ = self.model
        model = model_(embeddingMatrix)
        if X_test is not None:
            model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=self.n_epochs, batch_size=self.batch_size, verbose = 2)
        else:
            model.fit(X_train, y_train, 
                    epochs=self.n_epochs, batch_size=self.batch_size, verbose = 2)
        return model

    #def test():
        # Add test


def getEmbeddingMatrix(wordIndex=None, gloveDir=None, embedding_dim=None):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, embedding_dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    
    return embeddingMatrix
