import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

###########################################
#  Class to handle an LLM in inference mode
###########################################

class Inference(): 
    # -----------------------------------------------------------
    def __init__(self, model_path, quantized=False, peft=None, ollama=False) :
        ''' 
        Loads given model in inference mode 
        '''
        
        self.ollama = ollama
        if ollama:
            import ollama
            self.client = ollama.Client()
            self.model = model_path
            if quantized: print("Ignoring 'quantized' argument in ollama mode.")
            if peft: print("Ignoring 'peft' argument in ollama mode.")
            
        else :
            # set up quantization if needed
            if quantized :
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else :
               bnb_config = None

            self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                              dtype=torch.float16,
                                                              quantization_config=bnb_config,
                                                              device_map="auto")
                                                             
            # load fine-tuned weights if needed
            if peft: 
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, peft)

            # inference mode                                                     
            self.model.eval()

            # load tokenizer.        
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------- clean up GPU when deleting model
    def __del__(self):
        ''' 
        Clear GPU when done
        '''
        del self.tokenizer
        del self.model
        torch.cuda.empty_cache() 

    # ------------ generate completion for given messages ----------------
    def generate(self, messages):
        ''' 
        Generate completion for given messages
        '''  
        if self.ollama:
            response = self.client.chat(model = self.model,
                                        messages = messages,
                                        options = {"num_predict": 256}
                                        )
            gen_text = response.message.content
            return gen_text
            
        else:  
            # Tokenize and encode the prompt.
            input_ids = self.tokenizer.apply_chat_template(messages,
                                                           tokenize=True,
                                                           add_generation_prompt=True,
                                                           return_tensors="pt").to("cuda")
        
            # generate likely continuation (assistant answer)
            with torch.no_grad():
                gen_tokens = self.model.generate(input_ids,
                                                 max_new_tokens=256,
                                                 pad_token_id=self.tokenizer.eos_token_id
                                                )
            promptlen = len(input_ids[0])
            # decode obtained tokens back into text
            gen_text = self.tokenizer.decode(gen_tokens[0][promptlen:], skip_special_tokens=True)
            return gen_text

        
###########################################
#  Class to handle an LLM in Fine Tuning mode
###########################################

class FineTuning() :

    # ------------ load model and tokenizer -----------------
    def __init__(self, model_path, quantized=False):
        ''' 
        Loads given model in LoRa fine tuning mode 
        '''

        from peft import LoraConfig, get_peft_model

        # set up quantization if needed
        if quantized :
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else :
           bnb_config = None
        
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          dtype=torch.float16,
                                                          quantization_config=bnb_config,
                                                          device_map=None) # Can not load to GPU yet
            
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"

        # Add LoRa fine-tunable layers
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # Now, after adding PEFT layers, we can load to GPU.
        self.model = self.model.to("cuda")
        



    # ------------ tokenize dataset in batches of appropriate size -----------------
    def tokenize_dataset(self, dataset, prompts) :
        ''' 
        Tokenizes given dataset. 
        Returns a "datasets.Dataset" object suitable for trainer
        '''

        from datasets import Dataset

        # prepare to create tokenized and encoded version of the dataset 
        newDS = {"input_ids": [],
                 "labels": []}
        
        for example in dataset :
            # prepare messages for current example            
            msg = prompts.prepare_messages(example["input"],example["gold"])
            
            # convert messages to a whole text prompt
            text = self.tokenizer.apply_chat_template(msg, tokenize=False)

            # tokenize and encode text
            tokens = self.tokenizer(text,
                                    truncation=True,
                                    max_length=512,
                                    padding="max_length"
                                   )

            # mark padding tokens with -100 so the trainer ignores them
            labels = [-100 if tk == self.tokenizer.pad_token_id else tk for tk in tokens["input_ids"]]

            # add example to new dataset
            newDS["input_ids"].append(tokens["input_ids"])
            newDS["labels"].append(labels)

        # create and return tokenized+encoded dataset
        return Dataset.from_dict(newDS)


    # ------------ tokenize dataset in batches of appropriate size -----------------
    def train(self, train_dataset, val_dataset, outputdir) :
        ''' 
        Fine tuning model with given training and validation data.
        Save tuned weights in outputdir
        '''
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=outputdir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            eval_accumulation_steps=4,
            fp16=False,
            bf16=True,
            learning_rate=2e-5,
            num_train_epochs=10,
            eval_strategy="epoch",
            save_total_limit = 2,
            load_best_model_at_end=True,
            save_strategy = "epoch",
            logging_strategy="epoch",
            label_names=["labels"]
        )

        # Initialize the Trainer
        trainer = Trainer(
            model = self.model,
            args = training_args,
            eval_dataset = val_dataset,
            train_dataset = train_dataset
        )

        # do the fine tuning        
        trainer.train()
        
        # Save the fine-tuned model weighs (in outputdir)
        trainer.save_model()


