import torch
#from datasets import load_dataset
import math
import matplotlib.pyplot as plt
import pickle
import preprocessing
import argparse




def main(args):
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    num_epochs= args.epochs
    model_name = args.name



    model_checkpoint = args.model_filename
    model = preprocessing.create_model_MLM(model_checkpoint)
    tokenizer =preprocessing.create_tokenizer(model_checkpoint)

    #data_files = {"train": "swerick_data_long_train.pkl", "test": "swerick_data_long_test.pkl"}
    #swerick_dataset = load_dataset("pandas",data_files=data_files)
    #tokenized_datasets =preprocessing.tokenize_dataset(swerick_dataset,tokenizer)
    #lm_datasets = preprocessing.grouping_dataset(tokenized_datasets,chunk_size)

    with open("lm_dataset.pkl","rb") as fichier:
        lm_datasets=pickle.load(fichier)

    data_collator = preprocessing.data_collector_masking(tokenizer,0.15)
    logging_steps = len(lm_datasets["train"]) // batch_size

    trainer = preprocessing.create_trainer(model,model_name,batch_size,logging_steps,train_dataset=lm_datasets["train"],eval_dataset=lm_datasets["test"],data_collator=data_collator,tokenizer=tokenizer,num_epochs=num_epochs)
    trainer.train()
    model.save_pretrained("finetuning_trainer_total1")
    tokenizer.save_pretrained("finetuning_trainer_total1")

    train_losses=[]
    test_losses=[]
    for i in range(len(trainer.state.log_history)//2):
        train_losses.append(trainer.state.log_history[2*i]["loss"])
        test_losses.append(trainer.state.log_history[2*i+1]["eva_loss"])
    #eval_losses = trainer.state.log_history[\"eval_loss\"]

    #print(train_losses)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for the model")
    parser.add_argument("--name", type=str, default="finetuning_hugging_python", help="repository name")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)






