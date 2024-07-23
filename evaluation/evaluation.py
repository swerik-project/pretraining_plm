from transformers import Trainer
from transformers import TrainingArguments
import math
import torch
import subprocess
import preprocessing
from transformers import default_data_collator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def trainer_evaluation(model,dataloader):  
    logging_steps =892

    training_args = TrainingArguments(
        output_dir=f"{model.config.name_or_path}-imdb",
        per_device_eval_batch_size=64,
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01
        )


    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        eval_dataset=dataloader.dataset
    )
    return(trainer.evaluate())





def evaluation_task(model,dataloader,model_filename):
    print("Trainer evaluation....")
    #eval_results = trainer_evaluation(model,dataloader)
    #print(eval_results)
    #loss=next(iter(eval_results))
    #print(f">>> Perplexity: {math.exp(eval_results[loss]):.2f}")

    model.eval()
    model=model.to(device)
    losses=[]
    correct_predictions=0
    total_predictions=0
    for step, batch in enumerate(dataloader):
        batch={key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        indices_tokens_masked = torch.nonzero(batch["labels"] != -100, as_tuple=False)
        loss = outputs.loss
        losses.append(loss.repeat(dataloader.batch_size))
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        correct_predictions += torch.sum(
            torch.eq(batch["labels"][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]], 
                    predicted_token_ids[indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]])
        ).item()
        total_predictions += indices_tokens_masked.size(0)
       

    print("Manual perplexity...")
    losses = torch.cat(losses)
    losses = losses[: len(dataloader.dataset)]
    try:
       perplexity = math.exp(torch.mean(losses))
    except OverflowError:
       perplexity = float("inf")
    print(f" Perplexity: {perplexity}")


    print("Accuracy...")
   #     
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

def l2R_MLM_Crossentropy(model,dataset,tokenizer,max_length=128,batch_size=64):
    pll = 0
    batch_size=batch_size
    for i in  range(max_length):

        losses =[]
        eval_dataset_log =dataset.map(
            lambda examples: preprocessing.insert_special_masking_bis(examples,i,tokenizer),
            batched=True,
            remove_columns = dataset.column_names
        )
        eval_dataloader = preprocessing.create_dataloader(eval_dataset_log,batch_size,default_data_collator)
        for step, batch in enumerate(eval_dataloader):
            batch={key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                output=model(**batch)
            loss=output.loss
            losses.append(loss.repeat(eval_dataloader.batch_size))

        losses=torch.cat(losses)
        pll +=torch.mean(losses)

   
    print(f"Pseudo log perplexity hugging face: {pll}")
    print(f"Average pseudo log perplexity hugging_face: {pll/max_length}")
    return pll, pll/max_length
    
    



def regression_year(model_filename,data_path_train, tokenizer,data_path_test="/home/laurinemeier/swerick/evaluation/swerick_subsetdata_date_test.csv"):
    print("Year regression")

    print("training")
    command = [
    "python3",
    "train_regression.py",
    "--base_model2",
    model_filename,
    "--tokenizer2",
    tokenizer,
    "--model_filename2",
    "trained/regression_date"+model_filename[-6:],
    "--data_path",
    data_path_train

]

    # Exécuter la commande
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())


    print("comparing")
    command2= [
    "python3",
    "compare_models_regression.py",
    "--model_filename2",
    "trained/regression_date"+model_filename[-6:],
    "--data_path",
    data_path_test
    ]
    process = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())



    
    
def reform_scratch_classfication(model_filename,base_model1,base_model2,data_path_train, data_path_test,tokenizer1="KBLab/bert-base-swedish-cased",tokenizer2="KBLab/bert-base-swedish-cased"):

    print("Party alignement classification")
 
    print("training")
    command = [
    "python3",
    "train_binary_bert_base.py",
    "--model_filename",
    model_filename + base_model1[-6],
    "--base_model",
    base_model1,
    "--tokenizer",
    tokenizer1,
    "--data_path",
    data_path_train
]

    # Exécuter la commande
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    
        
    command = [
    "python3",
    "train_binary_bert_base.py",
    "--model_filename",
    model_filename+base_model2[-6:],
    "--base_model",
    base_model2,
    "--tokenizer",
    tokenizer2,
    "--data_path",
    data_path_train
]

    # Exécuter la commande
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    
    print("comparing")
    command2= [
    "python3",
    "compare_models.py",
    "--model_filename1",
    model_filename + base_model1[-6],
    "--model_filename2",
    model_filename +base_model2[-6:],
    "--tokenizer1",
    tokenizer1,
    "--tokenizer2",
    tokenizer2,
    "--data_path",
    data_path_test
    ]
    process = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    
    
    
    
