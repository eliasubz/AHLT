## --------------------------------------------------------------
## check whether a token is a stopword
def is_stopword(tk):
    # if it is not a Noun, Verb, adJective, or adveRb, then it is a stopword
    return tk.pos_ not in ['NOUN', 'VERB', 'ADJ', 'ADV']

## --------------------------------------------------------------
## check whether a token belongs to one of given entities
def is_entity(tk,entities):
    for e in entities :
        if entities[e]["start"] <= tk.idx and tk.idx+len(tk.text) <= entities[e]["end"]+1 :
            return e
    return None    

## --------------------------------------------------------------
## get position in the sentence of given token
def get_position(tree, tk):

    for i,t in enumerate(tree):
        if t == tk :
            return i
    # should never happen if the token is in the sentence
    return None
 
## --------------------------------------------------------------
## return the Lowest Common Subsumer of two nodes
def get_LCS(tree, n1, n2) :
    # check if n2 is ancestor of n1
    for tk1 in n1.ancestors :
        if tk1==n2: return n2
    # check if n1 is ancestor of n2
    for tk2 in n2.ancestors :
        if tk2==n1: return n1
    # get first common element in both ancestor lists
    for tk1 in n1.ancestors :
        for tk2 in n2.ancestors :
            if tk1 == tk2 :
                return tk1
     
    # (should never happen since tree root is always a common subsumer,
    # unless there are two sentences....)
    return None 

        
## --------------------------------------------------------------
## get upwards path from n1 to n2 (returns list of nodes
## from n1 to n2, including n1, upwards, excluding n2)
def get_up_path(n1,n2) :
    path = [n1]
    if n1==n2 :
        return path
        
    for n in n1.ancestors :
        if n == n2 :  # we found n2, stop
            break
        else :
            path.append(n)
    else:
        # we reached the end of the loop, n2 was not ancestor of n1
        return None
    
    return path
    
## --------------------------------------------------------------
## get downwards path from n1 to n2 (return list of nodes, downwards, excluding n1)
def get_down_path(n1,n2) :
    path = get_up_path(n2,n1)
    if path is not None: # if None, n1 was not ancestor of n2
        path.reverse()
    return path

## --------------------------------------------------------------
## get token heading the given sentence fragment (e.g. an entity span)
def get_fragment_head(tree, start, end) :
    # find which tokens overlap the fragment
    overlap = set() 
    for tk in tree:
        tk_start,tk_end = tk.idx, tk.idx+len(tk.text)    
        if tk_start <= start <= tk_end or tk_start <= end <= tk_end :
            overlap.add(tk)
                
    head = None
    if len(overlap)>0 :
        # find head node among those overlapping the entity
        for tk in overlap :
            if head is None: head = tk
            else: head = get_LCS(tree, head, tk)
                
        # if found LCS does not overlap the entity, the parsing was wrong, forget it.
        if head not in overlap :
            head = None
        
    return head     

# -----------------
# check pattern:  First verb at LCS or above, one entity is under its "nsubj" and the other under its "obj"
def check_pattern_verb_func(tree,entities,e1,e2):

   # get head token for each gold entity
   tkE1 = get_fragment_head(tree, entities[e1]['start'],entities[e1]['end'])
   tkE2 = get_fragment_head(tree, entities[e2]['start'],entities[e2]['end'])

   if tkE1 is None or tkE2 is None: 
       # we were not able to find the entity token
       return None

   # get LCS
   lcs = get_LCS(tree, tkE1,tkE2)
   if lcs is None : 
      # LCS not found
      return None
            
   # find first verb at or over LCS
   while lcs.pos_ != "VERB" and lcs.head != lcs :
      lcs = lcs.head

   if lcs.pos_ == "VERB":          
      path1 = get_up_path(tkE1,lcs)
      path2 = get_up_path(tkE2,lcs)
      func1 = path1[-1].dep_ if path1 else None
      func2 = path2[-1].dep_ if path2 else None

      if func1 is not None and func2 is not None :
         w = lcs.lemma_ + '_' + lcs.pos_
         return [w+":"+func1+"_"+func2]
   
   return None

# -----------------
# check pattern:  First verb at LCS or above
def check_pattern_verb_lcs(tree,entities,e1,e2):

   # get head token for each gold entity
   tkE1 = get_fragment_head(tree, entities[e1]['start'], entities[e1]['end'])
   tkE2 = get_fragment_head(tree, entities[e2]['start'], entities[e2]['end'])

   if tkE1 is None or tkE2 is None: 
       # we were not able to find the entity token
       return None

   # get LCS
   lcs = get_LCS(tree, tkE1,tkE2)
   if lcs is None : 
      # LCS not found
      return None
      
   # find first verb at or over LCS
   while (lcs.pos_ != "VERB") and (lcs.head != lcs) :
      lcs = lcs.head

   # build features to be added (lemma of the verb)
   if lcs.pos_ == "VERB":
      return [lcs.lemma_]
   
   return None

# -----------------
# check pattern: words in between
def check_pattern_wib(tree,entities,e1,e2):

   # get head token for each gold entity
   tkE1 = get_fragment_head(tree,entities[e1]['start'],entities[e1]['end'])
   tkE2 = get_fragment_head(tree,entities[e2]['start'],entities[e2]['end'])

   if tkE1 is None or tkE2 is None: 
       # we were not able to find the entity token
       return None

   # get actual start/end of both entities
   l1,r1 = entities[e1]['start'],entities[e1]['end']
   l2,r2 = entities[e2]['start'],entities[e2]['end']
  
   e1pos = get_position(tree, tkE1)
   e2pos = get_position(tree, tkE2)
   p = []
   for i in range(e1pos+1,e2pos) :
      tk = tree[i]
      if not is_stopword(tk):
          # get token span
          l,r = tk.idx, tk.idx+len(tk.text)    
          # if the token is in between both entities
          if r1 < l and r < l2:
              p.append("l="+tk.lemma_)
              p.append("lp="+tk.lemma_ + "_" + tk.pos_)
              # feature indicating the presence of an entity in between E1 and E2
              if is_entity(tk, entities) :
                  p.append("eib")
           
   if p: return p
   else: return None


# -----------------
# check pattern: words not in between both entities
def check_pattern_wout(tree,entities,e1,e2):

   # get head token for each gold entity
   tkE1 = get_fragment_head(tree, entities[e1]['start'], entities[e1]['end'])
   tkE2 = get_fragment_head(tree, entities[e2]['start'], entities[e2]['end'])

   if tkE1 is None or tkE2 is None: 
       # we were not able to find the entity token
       return None
       
   # get actual start/end of both entities
   l1,r1 = entities[e1]['start'],entities[e1]['end']
   l2,r2 = entities[e2]['start'],entities[e2]['end']
  
   p = []
   for tk in tree :
      if not is_stopword(tk):
         # get token span
         l,r = tk.idx, tk.idx+len(tk.text)    
         if r < l1 :
            # the token is before first entity
            p.append("lb1:"+tk.lemma_)
         elif r2 < l :
            # the token is after second entity
            p.append("la2:"+tk.lemma_)      
   if p: return p
   else: return None



# -----------------------------
# patterns to be used, and function computing them
patterns = {"pat_verb_lcs": check_pattern_verb_lcs,
            "pat_verb_func": check_pattern_verb_func,
            "pat_wib": check_pattern_wib,
            "pat_wout": check_pattern_wout
           }
