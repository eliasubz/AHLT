import json, copy

class Prompts:

    # ------------- load prompts from json file ----------------------
    def __init__(self, promptfile, fs_examples=[]) :
    
        with open(promptfile) as pf : 
            self.prompts = json.load(pf)
    
        for k in self.prompts : 
            self.prompts[k] = "\n".join(self.prompts[k])

        # First, system message
        self.messages = [{"role": "system", "content": self.prompts["sysprompt"]}]
     
        # add given few_shot examples to base prompt
        for ex in fs_examples :
           self.messages.append({"role": "user",
                                 "content": self.prompts["usrprompt"] 
                                            + "\nTEXT: "
                                            + ex['input']})
           self.messages.append({"role": "assistant", 
                                 "content": ex['gold']})

                        
    # ------------ prepare prompt messages for a particular example ----------------
    def prepare_messages(self, question, answer="") :
        # get base prompt
        msg = copy.deepcopy(self.messages)
        
        # add the text we want the anwer for, so the model will complete the response
        msg.append({"role": "user", 
                    "content": self.prompts["usrprompt"]
                               + "\nTEXT: "
                               + question})
                               
        if answer:
            msg.append({"role": "assistant", 
                        "content": answer})
        return msg
        
        
        
