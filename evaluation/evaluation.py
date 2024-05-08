from transformers import Trainer
from transformers import TrainingArguments
import math
import torch
import subprocess

from train_binary_bert import main as train_bert

from compare_models import main as compare

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


def party_gender_detection(model_filename) :

    print("Party classification")
 
    print("training")
    command = [
    "python3",
    "train_binary_bert.py",
    "--model_filename",
    "trained_party_classification",
    "--data_path",
    "swerick_subsetdata_party_train.csv",
    "--label_names",
    '"vänstern"',"Andra kammarens center","Andra kammarens frihandelsparti","Bondeförbundet","Centern (partigrupp 1873-1882)","Centern (partigrupp 1885-1887)","Centerpartiet","Det förenade högerpartiet","Ehrenheimska partiet","Folkpartiet","Folkpartiet (1895–1900)","Friesenska diskussionsklubben","Frihandelsvänliga centern","Frisinnade folkpartiet","Frisinnade försvarsvänner","Frisinnade landsföreningen","Första kammarens konservativa grupp","Första kammarens ministeriella grupp","Första kammarens minoritetsparti","Första kammarens moderata parti","Första kammarens nationella parti","Första kammarens protektionistiska parti","Gamla lantmannapartiet","Högerns riksdagsgrupp","Högerpartiet","Högerpartiet de konservativa","Jordbrukarnas fria grupp","Junkerpartiet","Kilbomspartiet","Kommunistiska partiet","Kristdemokraterna","Lantmanna- och borgarepartiet inom andrakammaren","Lantmannapartiet","Lantmannapartiets filial","Liberala riksdagspartiet","Liberala samlingspartiet","Liberalerna","Medborgerlig samling (1964–1968)","Miljöpartiet","Moderaterna","Nationella framstegspartiet","Ny demokrati","Nya centern (partigrupp 1883-1887)","Nya lantmannapartiet","Nyliberala partiet","Skånska partiet","Socialdemokraterna","Socialdemokratiska vänstergruppen","Socialistiska partiet","Stockholmsbänken","Sverigedemokraterna","Sveriges kommunistiska parti","Vänsterpartiet","borgmästarepartiet","frihandelsvänlig vilde","frisinnad vilde","högervilde","ministeriella partiet","partilös","politisk vilde","vänstervilde"
    #"man","woman"
]

    # Exécuter la commande
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    

    command = [
    "python3",
    "train_binary_bert.py",
    "--model_filename",
    "trained_hugging_face_party_classification"+ model_filename[-6:],
    "--base_model",
    model_filename,
    "--data_path",
    "swerick_subsetdata_party_train.csv",
    "--label_names",
    '"vänstern"',"Andra kammarens center","Andra kammarens frihandelsparti","Bondeförbundet","Centern (partigrupp 1873-1882)","Centern (partigrupp 1885-1887)","Centerpartiet","Det förenade högerpartiet","Ehrenheimska partiet","Folkpartiet","Folkpartiet (1895–1900)","Friesenska diskussionsklubben","Frihandelsvänliga centern","Frisinnade folkpartiet","Frisinnade försvarsvänner","Frisinnade landsföreningen","Första kammarens konservativa grupp","Första kammarens ministeriella grupp","Första kammarens minoritetsparti","Första kammarens moderata parti","Första kammarens nationella parti","Första kammarens protektionistiska parti","Gamla lantmannapartiet","Högerns riksdagsgrupp","Högerpartiet","Högerpartiet de konservativa","Jordbrukarnas fria grupp","Junkerpartiet","Kilbomspartiet","Kommunistiska partiet","Kristdemokraterna","Lantmanna- och borgarepartiet inom andrakammaren","Lantmannapartiet","Lantmannapartiets filial","Liberala riksdagspartiet","Liberala samlingspartiet","Liberalerna","Medborgerlig samling (1964–1968)","Miljöpartiet","Moderaterna","Nationella framstegspartiet","Ny demokrati","Nya centern (partigrupp 1883-1887)","Nya lantmannapartiet","Nyliberala partiet","Skånska partiet","Socialdemokraterna","Socialdemokratiska vänstergruppen","Socialistiska partiet","Stockholmsbänken","Sverigedemokraterna","Sveriges kommunistiska parti","Vänsterpartiet","borgmästarepartiet","frihandelsvänlig vilde","frisinnad vilde","högervilde","ministeriella partiet","partilös","politisk vilde","vänstervilde"
    #"man","woman"
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
    "trained_party_classification",
    "--model_filename2",
    "trained_hugging_face_party_classification" + model_filename[-6:],
    "--data_path",
    "swerick_subsetdata_party_test.csv",
    "--label_names",
    #"man","woman"
    '"vänstern"',"Andra kammarens center","Andra kammarens frihandelsparti","Bondeförbundet","Centern (partigrupp 1873-1882)","Centern (partigrupp 1885-1887)","Centerpartiet","Det förenade högerpartiet","Ehrenheimska partiet","Folkpartiet","Folkpartiet (1895–1900)","Friesenska diskussionsklubben","Frihandelsvänliga centern","Frisinnade folkpartiet","Frisinnade försvarsvänner","Frisinnade landsföreningen","Första kammarens konservativa grupp","Första kammarens ministeriella grupp","Första kammarens minoritetsparti","Första kammarens moderata parti","Första kammarens nationella parti","Första kammarens protektionistiska parti","Gamla lantmannapartiet","Högerns riksdagsgrupp","Högerpartiet","Högerpartiet de konservativa","Jordbrukarnas fria grupp","Junkerpartiet","Kilbomspartiet","Kommunistiska partiet","Kristdemokraterna","Lantmanna- och borgarepartiet inom andrakammaren","Lantmannapartiet","Lantmannapartiets filial","Liberala riksdagspartiet","Liberala samlingspartiet","Liberalerna","Medborgerlig samling (1964–1968)","Miljöpartiet","Moderaterna","Nationella framstegspartiet","Ny demokrati","Nya centern (partigrupp 1883-1887)","Nya lantmannapartiet","Nyliberala partiet","Skånska partiet","Socialdemokraterna","Socialdemokratiska vänstergruppen","Socialistiska partiet","Stockholmsbänken","Sverigedemokraterna","Sveriges kommunistiska parti","Vänsterpartiet","borgmästarepartiet","frihandelsvänlig vilde","frisinnad vilde","högervilde","ministeriella partiet","partilös","politisk vilde","vänstervilde"
    ]
    process = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())


def regression_year(model_filename,data_path):
    print("Year regression")

    print("training")
    command = [
    "python3",
    "train_regression.py",
    "--base_model2",
    model_filename,
    "--model_filename2",
    "trained/regression_date"+model_filename[-6:],
    "--data_path",
    data_path

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
    "swerick_subsetdata_date_test.csv"
    ]
    process = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
<<<<<<< HEAD:evaluation.py


def intro_classifaction(model_filename):

    print("intro classification")
 
    print("training")
    command = [
    "python3",
    "train_binary_bert.py",
    "--model_filename",
    "trained_intro_classification",
    "--data_path",
    "swerick_data_intro_train.csv"
]

    # Exécuter la commande
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
    

    command = [
    "python3",
    "train_binary_bert.py",
    "--model_filename",
    "trained_hugging_face_intro_classification"+ model_filename[-6:],
    "--base_model",
    model_filename,
    "--data_path",
    "swerick_data_intro_train.csv"
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
    "trained_intro_classification",
    "--model_filename2",
    "trained_hugging_face_intro_classification" + model_filename[-6:],
    "--data_path",
    "swerick_data_intro_test.csv"
    ]
    process = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("stdout:", stdout.decode())
    print("stderr:", stderr.decode())
=======
>>>>>>> fa650b69c61bd7cd9e1fc58dff2358592babade2:evaluation/evaluation.py
