import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import numpy as np

def train_w2v(data_training, data_validation):
    all_docstring = []
    all_tokens = []
    for dp in data_validation+data_training:
        docstr = dp["data"]["docstring"]
        all_tokens.append(dp["data"]["funcName"])
        for arg in dp["data"]["allArgs"]:
            if len(arg)>0:
                all_tokens.append(arg)
        for assign in dp["data"]["assigns"]:
            if len(assign)>0:
                all_tokens.append(assign)
                for i in re.split(r"[.,\(\)\[\]=\{\}\'\"]", assign):
                    if len(i)>0:
                        all_tokens.append(i)
        for ret in dp["data"]["returnStatements"]:
            if len(ret)>0:
                all_tokens.append(ret)
                for i in re.split(r"[,.:\"\' \+\-\*/%\(\)\[\]=\{\}]", ret):
                    if len(i)>0:
                        all_tokens.append(i)
        if docstr:
            for word in docstr:
                all_docstring.append(word)

    w2v_tokens = Word2Vec([all_tokens], min_count=1)
    w2v_docstring = Word2Vec([all_docstring], min_count=1)
    w2v_tokens.wv.save('w2v_tokens.wordvectors')
    w2v_docstring.wv.save('w2v_docstring.wordvectors')
    return

def train_w2v_labels(arg_train, arg_valid, ret_train, ret_valid):
    all_labels = []
    for dp in arg_train+arg_valid+ret_train+ret_valid:
        all_labels.append(dp["label"])
    w2v_labels = Word2Vec([all_labels], min_count=1)
    w2v_labels.wv.save('w2v_labels.wordvectors')
    return

def main():
    with open("arg_data_training.pkl", "rb") as f:
        arg_data_training = pickle.load(f)
    with open("arg_data_validation.pkl", "rb") as f:
        arg_data_validation = pickle.load(f)
    with open("ret_data_training.pkl", "rb") as f:
        ret_data_training = pickle.load(f)
    with open("ret_data_validation.pkl", "rb") as f:
        ret_data_validation = pickle.load(f)

    ## to be uncommented only on the first launch
    ## or if .wordvectors files are lost
    # train_w2v (ret_data_training, ret_data_validation)
    # train_w2v_labels(arg_data_training, arg_data_validation, 
    #       ret_data_training, ret_data_validation)

    # w2v_labels = KeyedVectors.load('w2v_labels.wordvectors', mmap='r')
    # w2v_tokens = KeyedVectors.load('w2v_tokens.wordvectors', mmap='r')
    w2v_docstring = KeyedVectors.load('w2v_docstring.wordvectors', mmap='r')

    arg_data_labels_training = []
    arg_data_labels_validation = []
    ret_data_labels_training = []
    ret_data_labels_validation = []
    arg_data_training_data = []

    for dp in arg_data_training:
        arg_data_labels_training.append(w2v_labels[dp["label"]])
        data = dp["data"]
        l = np.array(np.array(w2v_tokens[data["funcName"]].tolist, dtype=float), dtype=ndarray)
        np.append(l, w2v_tokens[data["argName"]].tolist)
        l.append([w2v_tokens[i] for i in data["allArgs"]])
        l.append([w2v_tokens[i] for i in data["assigns"]])
        l.append([w2v_tokens[i] for i in data["returnStatements"]])
        if data["docstring"]:
            l.append([w2v_docstring[i] for i in data["docstring"]])
        else:
            l.append([None])
        l.append(data["occurences"])
        arg_data_training_data.append(l)

    for dp in arg_data_validation:
        arg_data_labels_validation.append(w2v_labels[dp["label"]])

    for dp in ret_data_training:
        ret_data_labels_training.append(w2v_labels[dp["label"]])

    for dp in ret_data_validation:
        ret_data_labels_validation.append(w2v_labels[dp["label"]])

    x = np.array(arg_data_training_data)
    print (x.shape)
    # arg_data_labels_training = np.array(arg_data_labels_training)
    # arg_data_labels_validation = np.array(arg_data_labels_validation)
    # ret_data_labels_training = np.array(ret_data_labels_training)
    # ret_data_labels_validation = np.array(ret_data_labels_validation)

    # print (arg_data_labels_training.shape)
    # print (arg_data_labels_validation.shape)
    # print (ret_data_labels_training.shape)
    # print (ret_data_labels_validation.shape)

    return


if __name__ == "__main__":
    main()