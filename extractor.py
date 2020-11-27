# import nltk
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import pickle
from pprint import pprint
import os
import re
from gensim.models import Word2Vec

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
        words = []
        for sent in sents:
            for word in word_tokenize(sent):
                if word not in stopwords.words('english'):
                    words.append(word)
        return words
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

def main():
    output = []
    directory = r'data/validation'
    counter = 0
    for filename in os.listdir(directory):
        counter += 1
        try:
            filepath = os.path.join(directory, filename)
            output += process_file(filepath)
        except UnicodeDecodeError:
            pass
        except SyntaxError:
             pass

    (arg_data, ret_data) = make_objects(output)
    ## used for faster rerun
    # with open("arg_data.pkl", "wb") as f:
    #    pickle.dump(arg_data, f)

    # with open("ret_data.pkl", "wb") as f:
    #    pickle.dump(ret_data, f)
    # with open("arg_data.pkl", "rb") as f:
    #    arg_data = pickle.load(f)

    # with open("ret_data.pkl", "rb") as f:
    #    ret_data = pickle.load(f)
    all_words = []
    for obj in ret_data:
        docstr = obj["data"]["docstring"]
        if docstr:
            for word in docstr:
                all_words.append(docstr)
    word2vec = Word2Vec(all_words, min_count=1)

if __name__ == "__main__":
    main()