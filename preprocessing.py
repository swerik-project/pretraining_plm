import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_function(examples,tokenizer):
    result = tokenizer(examples["texte"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples,chunk_size):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def insert_random_mask(batch,data_collator,seed=42):
    torch.manual_seed(seed)
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


def create_model_MLM(model_checkpoint) :
    return AutoModelForMaskedLM.from_pretrained(model_checkpoint)

def create_tokenizer(model_checkpoint):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_dataset(dataset,tokenizer):

    return dataset.map(
      lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=dataset.column_names
)


def grouping_dataset(dataset,chunk_size) :
    return dataset.map( lambda examples: group_texts(examples,chunk_size), batched=True)

def data_collector_masking(tokenizer,mlm_proba):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_proba)

def compute_metrics(eval_preds):
    
    perplexity=0.1
    accuracy=0.3
    return {"Perplexity:": perplexity,"Accuracy :" : accuracy}

def create_trainer(model,model_name,batch_size,logging_steps,learning_rate=2e-5,decay=0.01,train_dataset=None,eval_dataset=None,data_collator=None,tokenizer=None,push_hub=False,num_epochs=None):
    training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    resume_from_checkpoint=True,
    overwrite_output_dir=True,
    save_strategy="epoch",
    save_total_limit=100,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=push_hub,
    fp16=True,
    logging_steps=logging_steps,
    logging_dir='./logs', 
    num_train_epochs=num_epochs
)
    
    return  Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
   # compute_metrics=compute_metrics
)

def create_deterministic_eval_dataset(dataset,data_collator):
    eval_dataset =dataset.map(
    lambda examples: insert_random_mask(examples,data_collator),
    batched=True,
    remove_columns=dataset.column_names,
)

    return eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
        }
    )

def create_dataloader(dataset,batch_size,collate_fct,shuffle=True):
    return DataLoader(dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=collate_fct
    )
def chunk_and_pad(examples,tokenizer,block_size):
        all_input_ids = []
        all_attention_masks = []
        all_word_ids=[]
 
        for text in examples['texte']:
            # Tokenize the text
            tokenized_inputs = tokenizer(text, truncation=False, return_attention_mask=True)
            input_ids = tokenized_inputs['input_ids']
            attention_mask = tokenized_inputs['attention_mask']
            word_ids = tokenized_inputs.word_ids()
            
            # Chunk into blocks of `block_size`
            for i in range(0, len(input_ids), block_size):
                chunk_input_ids = input_ids[i:min(i + block_size,len(input_ids))]
                chunk_attention_mask = attention_mask[i:min(i + block_size,len(input_ids))]
                chunk_word_ids = word_ids[i:min(i + block_size,len(input_ids))]
                
                # Pad the last chunk if necessary
                if len(chunk_input_ids) < block_size:
                    padding_length = block_size - len(chunk_input_ids)
                    chunk_input_ids = chunk_input_ids + [tokenizer.pad_token_id] * padding_length
                    chunk_attention_mask = chunk_attention_mask + [0] * padding_length
                    chunk_word_ids = chunk_word_ids + [-1] * padding_length
                
                all_input_ids.append(chunk_input_ids)
                all_attention_masks.append(chunk_attention_mask)
                all_word_ids.append(chunk_word_ids)
                
        labels =all_input_ids.copy()
    
        
        return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks,"labels":labels,"word_ids":all_word_ids}
    
    
def insert_special_masking_bis(batch, i,tokenizer):
    word_ids = batch["word_ids"]
    masked_input_id = batch["input_ids"].copy()
    attention_mask = batch["attention_mask"].copy()
    
    labels = np.full_like(masked_input_id, -100)
    
    for z, seq in enumerate(masked_input_id):
        if seq[i] == tokenizer.pad_token_id or seq[i] == tokenizer.sep_token_id:
            continue
        
        labels[z, i] = seq[i]
        masked_input_id[z][i] = tokenizer.mask_token_id
        future_token = [j for j, _ in enumerate(word_ids[z]) if word_ids[z][j] == word_ids[z][i] and j > i]
        
        for j in future_token:
            labels[z][j] = batch["input_ids"][z][j]
            masked_input_id[z][j] = tokenizer.mask_token_id
    
    return {
        "input_ids": masked_input_id,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    
def get_context_with_mask(example, token_id,tokenizer):
    masked_positions = [idx for idx, token in enumerate(example['input_ids']) if token == tokenizer.mask_token_id]
    print(masked_positions)
    for i in range(len(masked_positions)):
        if example['labels'][masked_positions[i]] == token_id:
            start_position = max(0, masked_positions[i-1]) if i > 0 else 0
            end_position = min(len(example["input_ids"]), masked_positions[i+1]) if i < len(masked_positions) - 1 else len(example["input_ids"])
            context = example["input_ids"][start_position + 1:end_position]
            return {'input_ids': context, 'attention_mask' : example['attention_mask'][start_position + 1:end_position],'labels': example['labels'][start_position + 1:end_position]}
    return {}# If no matching mask is found
  