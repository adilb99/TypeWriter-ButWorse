# import nltk
import ast
# from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint

def process_file(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    output = []    
    funcDefExtractor = FuncDefExtractor()
    argsAndReturnsExtractor = ArgsAndReturnsExtractor()
    bodyExtractor = BodyExtractor()

    funcDefExtractor.visit(tree)
    funcDefs = funcDefExtractor.returnStats()

    for funcDef in funcDefs:
        outObject = {
            "funcName": funcDef.name,
            "args": [],
            "returns": [],
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
            argObject = {
                "name": arg.arg,
                "type": arg.annotation
                }
            outObject["args"].append(argObject)


        # process return statements and return type label
        for ret in args_returns["returnStatement"]:
            retObject = {
                "statement": ret,
                "type": funcDef.returns
            }
            outObject["returns"].append(retObject)

        # processing assignments (e.g., x = 1)
        for assign in body["assigns"]:
            assObject = {
                "target": assign.targets,
                "value": assign.value
            }
            outObject["body"]["assigns"].append(assObject)

        # processing docstring
        outObject["body"]["docstring"] = body["docstring"]

        # processing number of occurences of certain data types
        outObject["body"]["numbers"] = body["numbers"]
        outObject["body"]["strings"] = body["strings"]
        outObject["body"]["booleans"] = body["booleans"]
        outObject["body"]["lists"] = body["lists"]
        outObject["body"]["tuples"] = body["tuples"]

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

def main():
    output = process_file("./data/test.py")
    
    print('funcName: ', output[0]["funcName"])
    print('args: ')
    for arg in output[0]["args"]:
        print('name: ', arg["name"], "; type: ", ast.dump(arg["type"]))
    
    print('returns: ')
    for ret in output[0]["returns"]:
        print("statement: ", ast.dump(ret["statement"]), "; type: ", ast.dump(ret["type"]))

    print('body: ')
    print('assigns: ')
    for assign in output[0]["body"]["assigns"]:
        print('targets: ')
        for tar in assign["target"]:
            print(ast.dump(tar))
        print('value: ', ast.dump(assign["value"]))

    print('docstring: ', ast.dump(output[0]["body"]["docstring"][0]))

    print('numbers: ', output[0]["body"]["numbers"])
    print('strings: ', output[0]["body"]["strings"])
    print('booleans: ', output[0]["body"]["booleans"])
    print('lists: ', output[0]["body"]["lists"])
    print('tuples: ', output[0]["body"]["tuples"])

if __name__ == "__main__":
    main()