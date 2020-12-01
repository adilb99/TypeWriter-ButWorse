import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import numpy as np

import pre_config

def populate_alphabet(data):
    all_chars = [";", pre_config.PAD_CHARACTER, pre_config.SEPARATOR, pre_config.SEPARATOR_MINOR]
    for dp in data:
        for ch in dp["data"]["funcName"]:
            all_chars.append(ch)
        for argname in dp["data"]["allArgs"]:
            for ch in argname:
                all_chars.append(ch)
        for assign in dp["data"]["assigns"]:
            for ch in assign:
                all_chars.append(ch)
        for retS in dp["data"]["returnStatements"]:
            for ch in retS:
                all_chars.append(ch)
    chars = sorted(list(set(all_chars)))
    return chars

def adjust_length(s, length):
    if len(s)<length:
        s += pre_config.PAD_CHARACTER * (length-len(s))
    elif len(s)>length:
        s = s[:length]
    return s

def makestr(dp_data):
    sep = pre_config.SEPARATOR
    sepm = pre_config.SEPARATOR_MINOR
    try:
        argname = adjust_length(dp_data["argName"], pre_config.fixlens["argName"])+sep
    except:
        argname = ""
    s = adjust_length(dp_data["funcName"], pre_config.fixlens["funcName"])+sep
    s += argname #sep is added on non-empty
    s += adjust_length(sepm.join(dp_data["allArgs"]), pre_config.fixlens["allArgs"])+sep
    s += adjust_length(sepm.join(dp_data["assigns"]), pre_config.fixlens["assigns"])+sep
    s += adjust_length(sepm.join(dp_data["returnStatements"]), pre_config.fixlens["returnStatements"])+sep
    return s

def makearray(data, chars):
    print (len(data))
    arrs = []
    jcount = 0
    for dp in data:
        jcount+=1
        if (jcount%1000==0):
            print (jcount)
        s = makestr(dp["data"])
        char_indices = dict((c, i) for i, c in enumerate(chars))

        win_size = pre_config.SLIDING_WINDOW_SIZE
        step = pre_config.STEP_SIZE
        sentences = []
        next_chars = []
        for i in range(0, len(s) - win_size, step):
            sentences.append(s[i: i + win_size])
            next_chars.append(s[i + win_size])

        x = np.zeros((len(sentences), win_size, len(chars)+1), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                try:
                    index = char_indices[char]
                except:
                    print("unknown char")
                    index = len(char_indices.keys())
                x[i, t, index] = 1
        arrs.append(x)
    print ("Stacking array. May take several minutes...")
    return np.stack(arrs, axis=0)

def docstr2array(data, keyvecs):
    length = pre_config.fixlens["docstring"]
    docstrs = []
    for dp in data:
        d = dp["data"]["docstring"]
        if d:
            words = []
            for sentence in d:
                words+=[w for w in sentence]
            if len(words)>length:
                words = words[:length]
            elif len(words)<length:
                words += [pre_config.PAD_CHARACTER for i in range(length - len(words))]
            arr = [keyvecs[w] for w in words]
            docstr_matrix = np.stack(arr)
        else:
            docstr_matrix = np.zeros((100, length), dtype = np.float32)
        docstrs.append (docstr_matrix)
    print ("Stacking array. May take several minutes...")
    return np.stack(docstrs, axis = 0)

def make_occurs(data):
    a = []
    for dp in data:
        occur = np.array(dp["data"]["occurences"], dtype=np.float32)
        a.append(occur)
    print ("Stacking array. May take several minutes...")
    return np.stack(a, axis=0)

def make_labels(data, all_labels):
    a = []
    length = pre_config.NUM_LABELS
    for dp in data:
        labels = np.zeros(length, dtype=np.bool)
        l = dp["label"]
        if l in all_labels:
            labels[all_labels.index(l)]=1
        else:
            labels[all_labels.index("OtherType")]=1
        a.append(labels)
    print ("Stacking array. May take several minutes...")
    return np.stack(a, axis=0)

            
def save_extracted():
    with open("arg_data_training.pkl", "rb") as f:
        arg_data_training = pickle.load(f)
    with open("arg_data_validation.pkl", "rb") as f:
        arg_data_validation = pickle.load(f)
    with open("ret_data_training.pkl", "rb") as f:
        ret_data_training = pickle.load(f)
    with open("ret_data_validation.pkl", "rb") as f:
        ret_data_validation = pickle.load(f)
    with open("all_labels.pkl", "rb") as f:
        all_labels = pickle.load(f)
    w2v_docstring = KeyedVectors.load('w2v_docstring.wordvectors', mmap='r')
    chars = populate_alphabet(arg_data_training+arg_data_validation)

    print ("********* Making main vecs")
    print ("arg_tr_arr")
    arg_tr_arr = makearray(arg_data_training, chars)
    print (arg_tr_arr.shape)
    with open ("numpy/arg_tr_main.npy", "wb") as f:
        np.save(f, arg_tr_arr)

    print ("arg_val_arr")
    arg_val_arr = makearray(arg_data_validation, chars)
    print (arg_val_arr.shape)
    with open ("numpy/arg_val_main.npy", "wb") as f:
        np.save(f, arg_val_arr)

    print ("ret_tr_arr")
    ret_tr_arr = makearray(ret_data_training, chars)
    print (ret_tr_arr.shape)
    with open ("numpy/ret_tr_main.npy", "wb") as f:
        np.save(f, ret_tr_arr)

    print ("ret_val_arr")
    ret_val_arr = makearray(ret_data_validation, chars)
    print (ret_val_arr.shape)
    with open ("numpy/ret_val_main.npy", "wb") as f:
        np.save(f, ret_val_arr)


    print ("********* Making docstr vecs")
    print ("arg_tr_arr")
    arg_tr_arr = docstr2array(arg_data_training, w2v_docstring)
    print (arg_tr_arr.shape)
    with open ("numpy/arg_tr_ds.npy", "wb") as f:
        np.save(f, arg_tr_arr)

    print ("arg_val_arr")
    arg_val_arr = docstr2array(arg_data_validation, w2v_docstring)
    print (arg_val_arr.shape)
    with open ("numpy/arg_val_ds.npy", "wb") as f:
        np.save(f, arg_val_arr)

    print ("ret_tr_arr")
    ret_tr_arr = docstr2array(ret_data_training, w2v_docstring)
    print (ret_tr_arr.shape)
    with open ("numpy/ret_tr_ds.npy", "wb") as f:
        np.save(f, ret_tr_arr)

    print ("ret_val_arr")
    ret_val_arr = docstr2array(ret_data_validation, w2v_docstring)
    print (ret_val_arr.shape)
    with open ("rnumpy/et_val_ds.npy", "wb") as f:
        np.save(f, ret_val_arr)

    print ("********* Making occurences vecs")
    print ("arg_tr_arr")
    arg_tr_arr = make_occurs(arg_data_training)
    print (arg_tr_arr.shape)
    with open ("numpy/arg_tr_oc.npy", "wb") as f:
        np.save(f, arg_tr_arr)

    print ("arg_val_arr")
    arg_val_arr = make_occurs(arg_data_validation)
    print (arg_val_arr.shape)
    with open ("numpy/arg_val_oc.npy", "wb") as f:
        np.save(f, arg_val_arr)

    print ("ret_tr_arr")
    ret_tr_arr = make_occurs(ret_data_training)
    print (ret_tr_arr.shape)
    with open ("numpy/ret_tr_oc.npy", "wb") as f:
        np.save(f, ret_tr_arr)

    print ("ret_val_arr")
    ret_val_arr = make_occurs(ret_data_validation)
    print (ret_val_arr.shape)
    with open ("numpy/ret_val_oc.npy", "wb") as f:
        np.save(f, ret_val_arr)

    print ("********* Making label vecs")
    print ("arg_tr_arr")
    arg_tr_arr = make_labels(arg_data_training, all_labels)
    print (arg_tr_arr.shape)
    with open ("numpy/arg_tr_y.npy", "wb") as f:
        np.save(f, arg_tr_arr)

    print ("arg_val_arr")
    arg_val_arr = make_labels(arg_data_validation, all_labels)
    print (arg_val_arr.shape)
    with open ("numpy/arg_val_y.npy", "wb") as f:
        np.save(f, arg_val_arr)

    print ("ret_tr_arr")
    ret_tr_arr = make_labels(ret_data_training, all_labels)
    print (ret_tr_arr.shape)
    with open ("numpy/ret_tr_y.npy", "wb") as f:
        np.save(f, ret_tr_arr)

    print ("ret_val_arr")
    ret_val_arr = make_labels(ret_data_validation, all_labels)
    print (ret_val_arr.shape)
    with open ("numpy/ret_val_y.npy", "wb") as f:
        np.save(f, ret_val_arr)


    return

def main():
    save_extracted()
    return

if __name__ == "__main__":
    main()