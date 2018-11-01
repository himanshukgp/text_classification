from nlp.classify import classifier

def main():
    a = classifier(lr=0.001,
                 n_folds=5, # Change default values of following to none and provide default in application
                 trainDataPath="/home/singh/Desktop/emocontext/starterkitdata/train.txt",
                 testDataPath="/home/singh/Desktop/emocontext/starterkitdata/devwithoutlabels.txt",
                 solutionPath="/home/singh/Desktop/emocontext/starterkitdata/test/test12.txt",
                 gloveDir="/home/singh/Desktop/emocontext/glove.6B/",
                 n_class=4,
                 max_nb_words=20000,
                 max_seq_length=100,
                 embedding_dim=100,
                 batch_size=200,
                 lstm_dim=128,
                 dropout=0.2,
                 n_epochs=40,
                 eval_kfold=False
                 )
    a()

if __name__ == '__main__':
    main()

'''
  "train_data_path" : "path/to/train.txt",
  "test_data_path" : "path/to/dev.txt",
  "solution_path" : "test.txt",
  "glove_dir" : "./",
  "num_folds" : 5,
  "num_classes" : 4,
  "max_nb_words" : 20000,
  "max_sequence_length" : 100,
  "embedding_dim" : 100,
  "batch_size" : 200,
  "lstm_dim" : 128,
  "learning_rate" : 0.003,
  "dropout" : 0.2,
  "num_epochs" : 75
'''