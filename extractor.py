# import nltk
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import pickle
import os
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import itertools
import numpy as np
import pre_config

def process_file(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    output = []    
    funcDefExtractor = FuncDefExtractor()

    funcDefExtractor.visit(tree)
    funcDefs = funcDefExtractor.returnStats()
    
    for funcDef in funcDefs:
        argsAndReturnsExtractor = ArgsAndReturnsExtractor()
        bodyExtractor = BodyExtractor()
        flag = True
        outObject = {
            "fileName": filename,
            "funcName": funcDef.name,
            "args": [],
            "returnType": "None",
            "returnStatements": [],
            "body": {
                "assigns": [], 
                "docstring": '',
                "numbers": 0,
                "strings": 0,
                "booleans": 0,
                "lists": 0,
                "tuples": 0
            }
        }

        argsAndReturnsExtractor.visit(funcDef)
        args_returns = argsAndReturnsExtractor.returnStats()

        for el in funcDef.body:
            bodyExtractor.visit(el)
        
        body = bodyExtractor.returnStats()
        
        # process all arguments
        for arg in args_returns["args"]:
            try:
                argObject = {
                    "name": arg.arg,
                    "type": ast.unparse(arg.annotation)
                    }
                outObject["args"].append(argObject)
            except:
                pass

        # process return statements and return type label
        for ret in args_returns["returnStatement"]:
            try:
                outObject["returnStatements"].append(ast.unparse(ret.value))
            except AttributeError:
                pass

        try:
            outObject["returnType"] = ast.unparse(funcDef.returns)
        except:
            flag = False

        # processing assignments (e.g., x = 1)
        identifiers = []
        for assign in body["assigns"]:
            try:
                identifiers += [ast.unparse(tar) for tar in assign.targets]
            except AttributeError:
                identifiers += [ast.unparse(assign.target)]
        outObject["body"]["assigns"] = identifiers

        # processing docstring
        outObject["body"]["docstring"] = ast.get_docstring(funcDef)
        
        # processing number of occurences of certain data types
        outObject["body"]["numbers"] = body["numbers"]
        outObject["body"]["strings"] = body["strings"]
        outObject["body"]["booleans"] = body["booleans"]
        outObject["body"]["lists"] = body["lists"]
        outObject["body"]["tuples"] = body["tuples"]

        if(flag):
            output.append(outObject)


    return output
    

class FuncDefExtractor(ast.NodeVisitor):
    def __init__(self):
            self.stats = []

    def returnStats(self):
        return self.stats

    def visit_FunctionDef(self, node):
        self.stats.append(node)
        self.generic_visit(node)

    


class ArgsAndReturnsExtractor(ast.NodeVisitor):
    def __init__(self):
            self.stats = { "args": [], "returnStatement": [] }

    def returnStats(self):
        return self.stats

    def visit_arg(self, node):
        self.stats["args"].append(node)
        self.generic_visit(node)

    def visit_Return(self, node):
        self.stats["returnStatement"].append(node)
        self.generic_visit(node)


class BodyExtractor(ast.NodeVisitor):
    def __init__(self):
            self.stats = { 
                "assigns": [], 
                "docstring": [],
                "numbers": 0,
                "strings": 0,
                "booleans": 0,
                "lists": 0,
                "tuples": 0
                 }

    def returnStats(self):
        return self.stats

    # local variables
    def visit_Assign(self, node):
        self.stats["assigns"].append(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.stats["assigns"].append(node)
        self.generic_visit(node)

    # doc string
    def visit_Expr(self, node):
        self.stats["docstring"].append(node)
        self.generic_visit(node)

    # ints and floats
    def visit_Num(self, node):
        self.stats["numbers"] += 1
        self.generic_visit(node)
    
    # strings
    def visit_Str(self, node):
        self.stats["strings"] += 1
        self.generic_visit(node)

    # booleans 
    def visit_Compare(self, node):
        self.stats["booleans"] += 1
        self.generic_visit(node)

    def visit_NameConstant(self, node):
        if(node.value == 'True' or node.value == 'False'):
            self.stats["booleans"] += 1
        self.generic_visit(node)

    def visit_List(self, node):
        self.stats["lists"] += 1
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.stats["tuples"] += 1
        self.generic_visit(node)

def preprocess_docstring(string):
    if (string):
        string = re.sub('[^a-zA-Z]', ' ', string)
        string = re.sub(r'\s+', ' ', string)
        sents = sent_tokenize(string)
        ret = []
        for sent in sents:
            new_sent = []
            for word in word_tokenize(sent):
                if word not in stopwords.words('english'):
                    new_sent.append(word)
            ret.append(new_sent)
        return ret
    else:
        return string


def make_objects(output):
    arg_data = []
    ret_data = []
    for func in output:
        docstr = preprocess_docstring(func["body"]["docstring"])
        for arg in func["args"]:
            arg_object = {
                "data": {
                    "fileName": func["fileName"],
                    "funcName": func["funcName"],
                    "argName": arg["name"],
                    "allArgs": [a["name"] for a in func["args"]],
                    "assigns": func["body"]["assigns"],
                    "returnStatements": func["returnStatements"],
                    "docstring": docstr,
                    "occurences": [func["body"]["numbers"], func["body"]["strings"], func["body"]["booleans"], func["body"]["lists"], func["body"]["tuples"]] 
                },
                "label": arg["type"]
            }

            arg_data.append(arg_object)

        ret_object = {
            "data": {
                "fileName": func["fileName"],
                "funcName": func["funcName"],
                "allArgs": [a["name"] for a in func["args"]],
                "assigns": func["body"]["assigns"],
                "returnStatements": func["returnStatements"],
                "docstring": docstr,
                "occurences": [func["body"]["numbers"], func["body"]["strings"], func["body"]["booleans"], func["body"]["lists"], func["body"]["tuples"]] 
            },
            "label": func["returnType"]
            }

        ret_data.append(ret_object)
    return (arg_data, ret_data)

def train_w2v(data_training, data_validation):
    all_docstring = [pre_config.PAD_CHARACTER]
    for dp in data_training+data_validation:
        docstr = dp["data"]["docstring"]
        if docstr:
            for sent in docstr:
                all_docstring.append(sent)
    w2v_docstring = Word2Vec(all_docstring, min_count=1)
    w2v_docstring.wv.save('w2v_docstring.wordvectors')
    return

def make_labels_list(arg_tr, arg_val, ret_tr, ret_val):
    all_labels=dict()
    for dp in arg_tr+arg_val+ret_tr+ret_val:
        label = dp["label"]
        if label not in all_labels:
            all_labels[label]=0
        else:
            all_labels[label]+=1
    #sort by descending frequency
    all_labels = dict(sorted(all_labels.items(), reverse=True, key=lambda item: item[1]))
    #slice the top 999 type labels
    all_labels = dict(itertools.islice(all_labels.items(), pre_config.NUM_LABELS-1))
    #add OtherTypes for everything else
    all_labels["OtherType"] = 0
    #save just the list
    all_labels = list(all_labels.keys())
    with open("all_labels.pkl", "wb") as f:
        pickle.dump(all_labels, f)

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

def makearray(s, chars):
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
    return x

def docstr2array(d, keyvecs):
    length = pre_config.fixlens["docstring"]
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
    return docstr_matrix

def save_info (data, prefix, chars, w2v, all_labels, limit):
    for i, dp in enumerate(data):
        if (limit != 0 and i>limit):
            break
        string = makestr(dp["data"])
        bodyinf = makearray(string, chars)
        doc = docstr2array(dp["data"]["docstring"], w2v)
        occurences = np.array(dp["data"]["occurences"], dtype=np.int16)
        length = pre_config.NUM_LABELS
        labels = np.zeros(length, dtype=np.bool)
        l = dp["label"]
        if l in all_labels:
            labels[all_labels.index(l)]=1
        else:
            labels[all_labels.index("OtherType")]=1
        direct = prefix.split("/")[0]
        ty = prefix.split("/")[1]
        fn = dp["data"]["fileName"].split("/")[-1].split("\\")[-1]
        filename = ty +"-"+ fn+"-"+dp["data"]["funcName"]
        if i == 1:
            print (bodyinf.shape)
            print (doc.shape)
            print (occurences.shape)
            print (labels.shape)
            os.makedirs("numpy/"+direct, exist_ok=True)
        with open ("numpy/"+direct+"/"+filename+"-body-"+str(i)+".npy", "wb") as f:
            np.save(f, bodyinf)
        with open ("numpy/"+direct+"/"+filename+"-doc-"+str(i)+".npy", "wb") as f:
            np.save(f, doc)
        with open ("numpy/"+direct+"/"+filename+"-occur-"+str(i)+".npy", "wb") as f:
            np.save(f, occurences)
        with open ("numpy/"+direct+"/"+filename+"-labels-"+str(i)+".npy", "wb") as f:
            np.save(f, labels)
    return


def main():
    ## Extracting data
    ## Uncomment this for first time use
    # output_tr = []
    # output_val = []
    # dir_training = r'data/training'
    # dir_validation = r'data/validation'
    # for filename in os.listdir(dir_training):
    #     try:
    #         filepath = os.path.join(dir_training, filename)
    #         output_tr += process_file(filepath)
    #     except UnicodeDecodeError:
    #         pass
    #     except SyntaxError:
    #          pass

    # for filename in os.listdir(dir_validation):
    #     try:
    #         filepath = os.path.join(dir_validation, filename)
    #         output_val += process_file(filepath)
    #     except UnicodeDecodeError:
    #         pass
    #     except SyntaxError:
    #          pass

    ## Reformatting the data
    ## Uncomment this for first time use
    # (arg_data_training, ret_data_training) = make_objects(output_tr)
    # (arg_data_validation, ret_data_validation) = make_objects(output_val)

    #  Opening existing pickles
    ## Comment this out for first time use
    with open("./pickles/arg_data_training.pkl", "rb") as f:
        arg_data_training = pickle.load(f)
    with open("./pickles/arg_data_validation.pkl", "rb") as f:
        arg_data_validation = pickle.load(f)
    with open("./pickles/ret_data_training.pkl", "rb") as f:
        ret_data_training = pickle.load(f)
    with open("./pickles/ret_data_validation.pkl", "rb") as f:
        ret_data_validation = pickle.load(f)

    ##  vectorizing docstring
    ## Uncomment this for first time use
    # print ("Training word2vec for docstring...")
    # train_w2v(ret_data_training, ret_data_validation)
    # print ("Making the list of all labels...")
    # make_labels_list(arg_data_training, arg_data_validation,
    #                  ret_data_training, ret_data_validation)

    ##Opening existing pickles
    ## Comment this out for first time use
    with open("./pickles/all_labels.pkl", "rb") as f:
        all_labels = pickle.load(f)
    w2v_docstring = KeyedVectors.load('w2v_docstring.wordvectors', mmap='r')

    # Populating the chars alphabet
    chars = populate_alphabet(arg_data_training+arg_data_validation)

    ## Create new pickles
    ## Uncomment this for first time use
    # with open("arg_data_training.pkl", "wb") as f:
    #     pickle.dump(arg_data_training, f)

    # with open("ret_data_training.pkl", "wb") as f:
    #     pickle.dump(ret_data_training, f)

    # with open("arg_data_validation.pkl", "wb") as f:
    #     pickle.dump(arg_data_validation, f)

    # with open("ret_data_validation.pkl", "wb") as f:
    #     pickle.dump(ret_data_validation, f)
    

    ## Saving extracted data
    ## Uncomment this for first time use
    # print ("training/arg")
    # save_info(arg_data_training, "training/arg", chars, w2v_docstring, all_labels, 25000)
    # print ("val/arg")
    # save_info(arg_data_validation, "validation/arg", chars, w2v_docstring, all_labels, 0)
    # print ("training/ret")
    # save_info(ret_data_training, "training/ret", chars, w2v_docstring, all_labels, 25000)
    # print ("validation/ret")
    # save_info(ret_data_validation, "validation/ret", chars, w2v_docstring, all_labels, 0)
    
    return

if __name__ == "__main__":
    main()