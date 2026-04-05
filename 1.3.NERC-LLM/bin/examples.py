import sys, random, re
from collections import Counter
from xml.dom.minidom import parse


class Examples() :

    # load dataset from xml file
    def __init__(self, xmlfile, task) :
       self.task = task
       self.data = []
       tree = parse(xmlfile)   
       # process each sentence in the file
       sentences = tree.getElementsByTagName("sentence")
       for s in sentences :
          sid = s.attributes["id"].value   # get sentence id
          stext = s.attributes["text"].value.strip()   # get sentence text
          entities = s.getElementsByTagName("entity") # get gold standard entities
          spans = []
          for e in entities :
             # for discontinuous entities, we only get the first span
             # (will not work, but there are few of them)
             eid = e.attributes["id"].value
             (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
             typ =  e.attributes["type"].value
             spans.append((int(start),int(end),typ,eid)) 

          pairs = s.getElementsByTagName("pair") # get gold standard pairs
          interactions = []
          for p in pairs :          
             pid = s.attributes["id"].value   # get pair id
             # ground truth
             ddi = p.attributes["ddi"].value
             if (ddi=="true") : dditype = p.attributes["type"].value
             else : dditype = "null"
             # target entities
             e1 = p.attributes["e1"].value
             e2 = p.attributes["e2"].value
             interactions.append((e1,e2,dditype,pid))

          # start with the last span to simplfy offset computation
          spans.sort(reverse=True)
          if task == "NER":
              # rewrite expected output as XML-marked drugs in the text.
              newtext = stext
              for start,end,typ,_ in spans :
                  newtext = newtext[:end+1] + f"</{typ}>" + newtext[end+1:]
                  newtext = newtext[:start] + f"<{typ}>" + newtext[start:]

              self.data.append({"id" : sid,
                                "input" : stext,
                                "gold" : newtext
                               })
                               
          elif task == "DDI":
              for e1,e2,dditype,pid in interactions :
                  newtext = stext
                  for start,end,typ,eid in spans :
                      if eid==e1 : drug = "[DRUG1]"
                      elif eid==e2 : drug = "[DRUG2]"
                      else: drug = "[DRUG_OTHER]"
                      newtext = newtext[:start] + drug + newtext[end+1:]

                  self.data.append({"id": pid,
                                    "sid" : sid,
                                    "e1" : e1,
                                    "e2" : e2,
                                    "input" : newtext,
                                    "gold" : dditype
                                   })
                                   
          print(f"Loaded {len(self.data)} examples for {task} from {xmlfile}", file=sys.stderr)
          
                                     
    # ------------ select given number of FS examples -----------------
    def select_examples(self, numFS=-1, balanced=False, strategy="random") :
        random.seed(12345)

        if balanced and self.task == "NER" :
            print(f"'balanced' not applicable for NER. Use strategy='diverse' instead.", file=sys.stderr)
            balanced = False

        if numFS == -1 :
            # return all dataset
            return self.data

        elif numFS == 0 :
            # for zero shot
            return []

        elif balanced :
            # return same amount of each class
            # only makes sense if there are few "gold" values (e.g. for DDI)

            # frequencies of each type
            types = Counter([x["gold"] for x in self.data])
            # balanced amout to expect
            n = max(1,numFS//len(types))

            # start with less frequent
            examples = []
            pending = len(types)
            for t in sorted(types, key=lambda x : types[x]) :
                filtered = [x for x in self.data if x["gold"]==t]
                if len(filtered) < n :
                    # not enough examples, add them all
                    examples.extend(filtered)
                    # distribute the missing among the classes not yet processed
                    pending -= 1
                    n += (n-len(filtered)+1)//pending
                    print(f"Selected {len(filtered)} examples for class {t}", file=sys.stderr)
                else :
                    examples.extend(random.sample(filtered, n))
                    print(f"Selected {n} examples for class {t}", file=sys.stderr)

            random.shuffle(examples)
            print(f"Selected {len(examples)} balanced examples", file=sys.stderr)
            return examples

        elif strategy == "diverse" and self.task == "NER":
            # Ensure all 4 drug types (drug, group, brand, drug_n) appear
            # in the selected examples. Greedily cover missing types first,
            # then fill remaining slots with a random selection.
            NER_TYPES = ["drug", "group", "brand", "drug_n"]

            def get_types(ex):
                return set(re.findall(r'<(drug|group|brand|drug_n)>', ex["gold"]))

            pool = list(self.data)
            random.shuffle(pool)

            selected = []
            covered  = set()

            # first pass: one example per type, preferring rarest type first
            type_freq = Counter(t for ex in pool for t in get_types(ex))
            for typ in sorted(NER_TYPES, key=lambda t: type_freq.get(t, 0)):
                if typ not in covered and len(selected) < numFS:
                    candidates = [ex for ex in pool if typ in get_types(ex)
                                  and ex not in selected]
                    if candidates:
                        pick = candidates[0]
                        selected.append(pick)
                        covered |= get_types(pick)

            # second pass: fill remaining slots randomly
            remaining = [ex for ex in pool if ex not in selected]
            still_needed = numFS - len(selected)
            if still_needed > 0 and remaining:
                selected.extend(random.sample(remaining,
                                              min(still_needed, len(remaining))))

            random.shuffle(selected)
            covered_str = ", ".join(sorted(covered))
            print(f"Selected {len(selected)} diverse examples covering: {covered_str}", file=sys.stderr)
            return selected[:numFS]

        else:
            # return random selection
            print(f"Selected {numFS} random examples", file=sys.stderr)
            return random.sample(self.data, numFS)
            
            
    
    # ------------ convert xml marks to format expected by the evaluator ----------------
    def NER_eval_format(self, ex, text):
       xmlopen = "<drug>|<group>|<brand>|<drug_n>"
       
       original = text
       fmt = []
       p = re.search(xmlopen, text)
       while p is not None :
           os, oe = p.span()
           tag = p.group()[1:-1] # remove < >
           taglen = oe-os+1

           p = re.search(f"</{tag}>", text)  # find closing tag 

           ok = False
           if p is not None:
              cs, ce = p.span()       
              if f"<{tag}>" not in text[oe:cs]:
                 # make sure closing tag is the closest
                 drug = text[oe:cs]
                 ok = True
                 
           if not ok :
              print(f"Missing closing {tag} in: {original}", file=sys.stderr)
              
           text = text[:os]+text[oe:]  # remove opening mark
           if ok:
              text = text[:cs]+text[ce:]  # remove closing mark
              fmt.append(f"{ex['id']}|{os}-{cs-taglen}|{drug}|{tag}")
              
           p = re.search(xmlopen, text)
           
       return fmt       

    # ------------ convert example to format expected by the evaluator ----------------
    def DDI_eval_format(self, ex, text):
       p = text.find("\n")
       if p>0: text = text[:p]
       if text not in ["null", "none"] :
          return f"{ex['sid']}|{ex['e1']}|{ex['e2']}|{text}" 
       else:
          return ""



    # ------------ convert example to format expected by the evaluator ----------------
    def eval_format(self, ex, text):
        if self.task == "NER":
            return self.NER_eval_format(ex,text)
        elif self.task == "DDI":
            return self.DDI_eval_format(ex,text)

