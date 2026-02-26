
import os, sys
import json

# ------------------------------------------
class DrugIndex() :
    def __init__(self, filename=None, resources=None) :
        
        if filename is not None:
            with open(filename) as f :
                self.tree = json.load(f)

        elif resources is not None :
            self.tree = {}
            print("Collecting drugs from HSDB")
            with open(os.path.join(resources,"HSDB.txt")) as h :
                n = 0
                for x in h.readlines() :
                    tks = x.strip().lower().split()
                    self.add_drug(self.tree, tks, "drug")
                    n += 1
                    if n%11==0 : print(f"{n} lines processed.        \r", end="")

            print("Collecting drugs from DrugBank")
            with open(os.path.join(resources,"DrugBank.txt")) as h :
                for x in h.readlines() :
                    (n,t) = x.strip().lower().split("|")
                    tks = n.split()
                    self.add_drug(self.tree, tks, t)

            print("Collecting drugs from drugs-train")
            with open(os.path.join(resources,"drugs-train.txt")) as h :
                for x in h.readlines() :
                    (_,_,n,t) = x.strip().lower().split("|")
                    tks = n.split()
                    self.add_drug(self.tree, tks, t)
        else :
            print("Error: either a filename or a resources file was expected")
            sys.exit(1)


    # ------------------------------------------
    def add_drug(self, node, tks, kind) :
        if tks[0] not in node :
            node[tks[0]] = {}

        if len(tks)==1 :
            node[tks[0]]["END"] = kind
        else :
            self.add_drug(node[tks[0]], tks[1:], kind)

    # ------------------------------------------
    def search_drug(self, tree, tks, i) :
        
        if tks[i] in tree :
            kind = None
            if i<len(tks)-1 :
                kind, end = self.search_drug(tree[tks[i]], tks, i+1)
                
            if kind is not None :
                return kind, end
            elif "END" in tree[tks[i]] :
                return tree[tks[i]]["END"], i

        # not in tree, or "END" not found for given fragment
        return None, 0

    # ------------------------------------------
    def find_drug(self, tks, i) :
        return self.search_drug(self.tree, [t.text.lower() for t in tks], i)

    # ------------------------------------------
    def dump(self, file=sys.stdout) :
        json.dump(self.tree, file, indent=3, ensure_ascii=False)
        
        
             
if __name__ == "__main__" :

    outfile = sys.argv[1]

    # create a new index and dump it to file
    import paths
    drugs = DrugIndex(resources=paths.RESOURCES)
    with open(outfile, "w") as of:
       drugs.dump(file=of)
      
      
