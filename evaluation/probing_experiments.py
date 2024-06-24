import sys
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        
class ScalarMixModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_labels):
        super(ScalarMixModel, self).__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden_size, num_labels) 
    
    def forward(self, layers,upto_layers=None):
        batch_size, num_layers, seq_length, hidden_size = layers.size()
        if upto_layers is not None:
            num_layers = upto_layers +1
        norm_weights = torch.nn.functional.softmax(self.weights[:num_layers], dim=0)
        norm_weights = norm_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        mixed_representation = torch.sum(norm_weights * layers[:,:num_layers,:,:], dim=1)
        scalar = self.gamma * mixed_representation
        cls_representation = scalar[:, 0, :]
        return self.classifier(cls_representation)
    
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            row['content'],
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(row['tag'], dtype=torch.long)
        }
        

            
class GridLocProbeExperiment:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_data()
        self.setup_model()
        set_seed(self.config.seed)
        
        
    tokenizer_class = BertTokenizer
    num_layers = 13
    hidden_size = 768 
    
    def reset_weights(self, model):
        if isinstance(model, nn.Module):
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            # Also reset parameters of ScalarMixModel itself
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            # Explicitly reset the weights and classifier
            if hasattr(model, 'weights'):
                torch.nn.init.zeros_(model.weights)
            if hasattr(model, 'classifier'):
                torch.nn.init.xavier_uniform_(model.classifier.weight)
                if model.classifier.bias is not None:
                    torch.nn.init.zeros_(model.classifier.bias)
            
            
    def setup_data(self):
        self.bert_tokenizer = self.tokenizer_class.from_pretrained(self.config.tokenizer)
        self.dataset_train = CustomDataset(self.config.data_path_train, self.bert_tokenizer)
        self.dataset_test = CustomDataset(self.config.data_path_test, self.bert_tokenizer)
        self.dataset_valid= CustomDataset(self.config.data_path_valid, self.bert_tokenizer)

    def setup_model(self):
        self.bert_model = BertModel.from_pretrained(self.config.bert_version).to(self.device)
        # Use a local BertConfig to avoid connection errors
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.model = ScalarMixModel (self.num_layers,self.hidden_size, 2).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        optimizers = {
            'adam': torch.optim.Adam,
        }
        self.optimizer = optimizers[self.config.optimizer](self.model.parameters(), lr=self.config.learning_rate)

    def probe(self):
        train_loader = DataLoader(self.dataset_train, batch_size=self.config.batch_size, shuffle=True)
        valid_loader = DataLoader(self.dataset_valid, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(self.dataset_test, batch_size=self.config.batch_size, shuffle=False)
 
        self.reset_weights(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate) 
        print(self.model.weights)
        for i in range(self.config.epochs):
            self.train_epoch(train_loader,upto_layers=self.num_layers - 1)
            test_metrics = self.evaluate(test_loader,upto_layers=self.num_layers - 1)
            print(self.model.weights)
            print(f"Epoch {i}: Test Accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}")



    def cumulative_probe(self):
        
        train_loader = DataLoader(self.dataset_train, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(self.dataset_test, batch_size=self.config.batch_size, shuffle=False)

        f1_scores = []

        for upto_layer in range(self.num_layers):
            self.reset_weights(self.model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            print(self.model.weights)
            for epoch in range(self.config.epochs):
                self.train_epoch(train_loader, upto_layers=upto_layer)
            test_metrics = self.evaluate(test_loader, upto_layers=upto_layer)
            f1_scores.append(test_metrics['f1'])
            print(self.model.weights)
            print(f"Layer {upto_layer}: Test Accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}")

        differential_scores = [f1_scores[0]]
        for i in range(1, len(f1_scores)):
            differential_scores.append(f1_scores[i] - f1_scores[i - 1])

        return f1_scores,differential_scores
        
    def train_epoch(self, dataloader,upto_layers=None):
        self.model.train()

        for i, batch in enumerate(tqdm(dataloader, file=sys.stderr)):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            label = batch['label'].to(self.device)
           

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'], dim=1).detach().to(self.device)
            output = self.model(bert_full_hidden,upto_layers=upto_layers)
            loss = self.loss_function(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, dataloader,upto_layers=None):
        
        self.model.eval()

        correct = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dataloader:

                gold = batch['label'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
                bert_full_hidden = torch.stack(bert_output['hidden_states'], dim=1).detach().to(self.device)
                output = self.model(bert_full_hidden,upto_layers=upto_layers)

                pred = output.argmax(dim=1)
                correct += torch.sum(gold == pred).item()

                y_true.append(gold.cpu())
                y_pred.append(pred.cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        accuracy = correct / len(dataloader.dataset)
        f1 = f1_score(y_true, y_pred, average='macro')
        return {'accuracy': accuracy, 'f1': f1}
    
    
    
    

    
    