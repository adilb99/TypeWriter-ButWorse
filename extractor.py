# import nltk
import ast
# from nltk.tokenize import sent_tokenize, word_tokenize
from pprint import pprint

def build_tree(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
    
    return tree

# class Analyzer(ast.NodeVisitor):
#     def __init__(self):
#             self.stats = {"import": [], "from": []}



#     def report(self):
#         pprint(self.stats)


def main():
    tree = build_tree("./data/test.py")
    print(ast.dump(tree))

if __name__ == "__main__":
    main()