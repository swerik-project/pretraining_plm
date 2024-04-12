import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer

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

def insert_random_mask(batch,data_collator):
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
      lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=["texte", "protocole"]
)


def grouping_dataset(dataset,chunk_size) :
    return dataset.map( lambda examples: group_texts(examples,chunk_size), batched=True)

def data_collector_masking(tokenizer,mlm_proba):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_proba)

def compute_metrics(eval_preds):
    
    losses=[]
    correct_predictions=0
    total_predictions=0
    print(eval_preds)
    for step, batch in enumerate(eval_preds):
        batch={key: value.to(device) for key, value in batch.items()}
        indices_tokens_masked = torch.nonzero(batch["labels"] != -100, as_tuple=False)
        loss = batch.loss
        losses.append(loss.repeat(64))
        predicted_token_ids = torch.argmax(batch.logits, dim=-1)
        correct_predictions += torch.sum(
            torch.eq(batch["labels"][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]], 
                    predicted_token_ids[indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]])
        ).item()
        total_predictions += indices_tokens_masked.size(0)
    losses = torch.cat(losses)
    losses = losses[: len(eval_preds)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
       perplexity = float("inf")
    accuracy = correct_predictions / total_predictions
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
    model=model.to("cpu"),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
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
