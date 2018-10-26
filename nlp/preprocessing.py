import numpy as np
import json, argparse, os
import re
import io
import sys

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def preprocessData(dataFilePath, mode="train"):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4])
            
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            indices.append(int(line[0]))
            conversations.append(conv.lower())
    
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')

