"""
Train BERT-based classifier
"""
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from trainerlog import get_logger
from bidict import bidict
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import argparse
import logging
import torch
import os

LOGGER = get_logger("train-bert")

def encode(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ix, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
                            row['content'],                      
                            add_special_tokens = True,
                            max_length = 512,
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['tag'].tolist())

    return input_ids, attention_masks, labels

#def BertForRegression(base_model,device):
  
  #class BertRegressor(nn.Module):
    
  #  def __init__(self, drop_rate=0.2):
        
     #   super(BertRegressor, self).__init__()
     #   D_in, D_out = 768, 1
        
     #   self.bert = \
            #       AutoModel.from_pretrained(base_model)
      #  self.regressor = nn.Sequential(
        #    nn.Dropout(drop_rate),
        #    nn.Linear(D_in, D_out))
   # def forward(self, input_ids, attention_masks):
        
       # outputs = self.bert(input_ids, attention_masks)
       # class_label_output = outputs[1]
        #outputs = self.regressor(class_label_output)
        #return outputs

 # return BertRegressor(drop_rate=0.2).to(device)

def r2_score(outputs, labels):
    predictions = outputs.logits.squeeze()
    labels_mean = torch.mean(labels.float())
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - predictions) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate(model, loader,loss_function):
    loss, valid_r2 = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        labels = batch[2].float().to(args.device)
        output = model(input_ids,token_type_ids=None,attention_mask=input_mask,labels=labels)
        loss +=output.loss.item()
        r2 = r2_score(output, labels)
        valid_r2.append(r2.item())
        
    r2 = torch.mean(torch.tensor(valid_r2))
    return loss, r2


def main(args):
    os.makedirs(args.model_filename1, exist_ok=True)
    os.makedirs(args.model_filename2, exist_ok=True)
    logging.basicConfig(filename=f'{args.model_filename1}/training.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    df = pd.read_csv(f'{args.data_path}')
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    df = df[df["content"].notnull()]
    #label_names = args.label_names
    #if label_names is None:
       # label_names = sorted(list(set(df["tag"])))
    #label_dict = {ix: name for ix, name in enumerate(label_names)}
    #df["tag"] = [bidict(label_dict).inv[tag] for tag in df["tag"]]

    LOGGER.info("Load and save tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.model_filename1)

    
    LOGGER.info("Preprocess datasets...")
    input_ids, attention_masks, labels = encode(df, tokenizer)

    LOGGER.info(f"Labels: {labels}")

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size  = int(args.train_ratio * len(dataset))
    val_size    = int(args.valid_ratio * len(dataset))
    test_size   = len(dataset) - train_size - val_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )
    print("length train loader :", train_size)
    valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )
    print("length valid loader :", val_size)
    # Not used atm
    test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )

    LOGGER.debug("Define model...")
    print(args.base_model1)
    model1 =AutoModelForSequenceClassification.from_pretrained(
        args.base_model1,
        num_labels=1).to(args.device)
    print(args.base_model2)
    model2 =AutoModelForSequenceClassification.from_pretrained(
        args.base_model2,
        num_labels=1).to(args.device)
    # Initialize optimizer
    loss_function = nn.MSELoss()
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model1.parameters()), lr=args.learning_rate)
    
    optimizer2= torch.optim.Adam(
        filter(lambda p: p.requires_grad, model2.parameters()), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.n_epochs
    num_warmup_steps = num_training_steps // 10

    # Linear warmup and step decay
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer = optimizer1,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
        )

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer = optimizer2,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
        )
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(args.n_epochs):
        LOGGER.train(f"Epoch {epoch} starts!")
        train_loss = 0
        model1.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            model1.zero_grad()  
            input_ids = batch[0].to(args.device)
            input_mask = batch[1].to(args.device)
            labels = batch[2].float().to(args.device)
            output = model1(input_ids,token_type_ids=None,attention_mask =input_mask,labels=labels)
            loss = output.loss
            train_loss += loss.item()

            loss.backward()
            optimizer1.step()
            scheduler1.step()
        
        # Evaluation
        valid_loss, valid_r2 = evaluate(model1, valid_loader,loss_function)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_loss_avg = train_loss * args.batch_size / len(train_loader)
        valid_loss_avg = valid_loss * args.batch_size / len(valid_loader)

        LOGGER.train(f'Training Loss: {train_loss_avg:.3f}')
        LOGGER.train(f'Validation Loss: {valid_loss_avg:.3f}')
        LOGGER.train(f'Validation r2: {valid_r2}')

        # Store best model

        if valid_loss < best_valid_loss:
            LOGGER.info("Best validation loss so far")
            best_valid_loss = valid_loss
            model1.save_pretrained(args.model_filename1)
            tokenizer.save_pretrained(args.model_filename1)
        else:
            LOGGER.info("Not the best validation loss so far")
    del model1
    del optimizer1
    del scheduler1
    print("new model")
    print(args.base_model2)
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(args.n_epochs):
        LOGGER.train(f"Epoch {epoch} starts!")
        train_loss = 0
        model2.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            model2.zero_grad()  
            input_ids = batch[0].to(args.device)
            input_mask = batch[1].to(args.device)
            labels = batch[2].float().to(args.device)
            output = model2(input_ids,token_type_ids=None,attention_mask =input_mask,labels=labels)
            loss = output.loss
            train_loss += loss.item()

            loss.backward()
            optimizer2.step()
            scheduler2.step()
        
        # Evaluation
        valid_loss, valid_r2 = evaluate(model2, valid_loader,loss_function)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_loss_avg = train_loss * args.batch_size / len(train_loader)
        valid_loss_avg = valid_loss * args.batch_size / len(valid_loader)

        LOGGER.train(f'Training Loss: {train_loss_avg:.3f}')
        LOGGER.train(f'Validation Loss: {valid_loss_avg:.3f}')
        LOGGER.train(f'Validation r2: {valid_r2}')

        # Store best model

        if valid_loss < best_valid_loss:
            LOGGER.info("Best validation loss so far")
            best_valid_loss = valid_loss
            model2.save_pretrained(args.model_filename2)
            tokenizer.save_pretrained(args.model_filename2)
        else:
            LOGGER.info("Not the best validation loss so far")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename1", type=str, default="trained/regression_date", help="Save location for the model")
    parser.add_argument("--model_filename2", type=str, default="trained/regression_date_hugging_face", help="Save location for the model")
    parser.add_argument("--base_model1", type=str, default="KBLab/bert-base-swedish-cased", help="Base model that the model is initialized from")
    parser.add_argument("--base_model2", type=str, default="KBLab/bert-base-swedish-cased", help="Base model that the model is initialized from")
    parser.add_argument("--tokenizer", type=str, default="KBLab/bert-base-swedish-cased", help="Which tokenizer to use; accepts local and huggingface tokenizers.")
    #parser.add_argument("--label_names", type=str, nargs="+", default=None, help="A list of label names to be used in the classifier. If None, takes class names from 'tag' column in the data.")
    parser.add_argument("--data_path", type=str, default="swerick_subsetdata_date_train1000.csv", help="Training data as a .CSV file. Needs to have 'content' (X) and 'tag' (Y) columns.")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use for training. Use 'cpu' for CPU.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.00002)
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of data used for training")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="Proportion of data used for validation. Test set is what remains after train and valid splits")
    args = parser.parse_args()
    main(args)
