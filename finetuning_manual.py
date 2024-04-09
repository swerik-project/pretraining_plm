from transformers import AutoModelForMaskedLM
import torch
from datasets import load_dataset
import math
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import preprocessing
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def insert_random_mask(batch,data_collator):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def to_device(batch):
    return {key: value.to(device) for key, value in batch.items()}

def main(args):
    model = preprocessing.create_model_MLM(args.model_checkpoint)
    tokenizer =preprocessing.create_tokenizer(args.model_checkpoint)
    data_files = {"train": args.train, "test": args.test}
    swerick_dataset = load_dataset("pandas",data_files=data_files)
    tokenized_datasets =preprocessing.tokenize_dataset(swerick_dataset,tokenizer)
    lm_datasets = preprocessing.grouping_dataset(tokenized_datasets,args.chunk_size)
    data_collator = preprocessing.data_collector_masking(tokenizer,args.mlml)
    lm_dataset_bis = lm_datasets.remove_columns(["word_ids","token_type_ids"])
    eval_dataset = preprocessing.create_deterministic_eval_dataset(lm_dataset_bis["test"],data_collator)
    train_dataloader = preprocessing.create_dataloader(lm_dataset_bis["train"],args.batch_size,data_collator)
    eval_dataloader = preprocessing.create_dataloader(eval_dataset,args.batch_size,default_data_collator)

    for batch in train_dataloader:
        batch = to_device(batch)
    for batch in eval_dataloader:
        batch = to_device(batch)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))

    losses_train=[]
    losses_test=[]
    #train_dataloader = get_dataloader()
    for epoch in range(args.num_train_epochs):
        # Training
        model.train()
        total_loss_train = 0.0 
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        losses_train.append(total_loss_train/len(train_dataloader))
    

        # Evaluation
        model.eval()
        losses=[]
        total_loss_eval=0.0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.repeat(args.batch_size))
            total_loss_eval +=loss.item()


        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

            losses_test.append(total_loss_eval/len(eval_dataloader))


            print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

            print("losses_test",losses_test)

        print("epoch",args.num_train_epochs)


    plt.plot(range(args.num_train_epochs),losses_train,label="train Loss")

    plt.plot(range(args.num_train_epochs),losses_test,label="test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    file_path = "finetuning_manual"
    model.save_pretrained(file_path)
    tokenizer.save_pretrained(file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_checkpoint", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for the model")
    parser.add_argument("--train", type=str, default="swerick_data_long_train", help="train data str")
    parser.add_argument("--test", type=str, default="swerick_data_long_test", help="test_data str")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mlm", type=float, default=0.15)
    args = parser.parse_args()

    main(args)


                        


