# import nltk
import ast
# from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint

def process_file(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
    
    funcDefExtractor = FuncDefExtractor()
    funcDefExtractor.visit(tree)

    funcDefs = funcDefExtractor.returnStats()

    output = []

    for funcDef in funcDefs:
        visitor = Visitor()
        visitor.visit(funcDef)
        stats = visitor.returnStats()
        outObject = {
            "name": funcDef.name, 
            "returnTypes": ast.dump(funcDef.returns),
            "args": stats["args"],
            "returnStatement": stats["returnStatement"]
            }
        
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

    


class Visitor(ast.NodeVisitor):
    def __init__(self):
            self.stats = { "args": [], "returnStatement": [] }

    def returnStats(self):
        return self.stats

    def visit_arg(self, node):
        self.stats["args"].append(ast.dump(node))
        self.generic_visit(node)

    def visit_Return(self, node):
        self.stats["returnStatement"].append(ast.dump(node))
        self.generic_visit(node)



def main():
    output = process_file("./data/test.py")
    print(output)


if __name__ == "__main__":
    main()