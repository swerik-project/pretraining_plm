import matplotlib.pyplot as plt
import numpy as np
import os
import re
from transformers import AutoModelForMaskedLM
import torch
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_weights(model):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    return weights


def comparaison_weights(model1,model2):
    weights_kb = get_model_weights(model1)
    weights_finetuned = get_model_weights(model2)

    weight_diffs = {}
    for key in weights_kb.keys():
        weight_diffs[key] = weights_finetuned[key] - weights_kb[key]
        
    norms = [np.linalg.norm(weight_diffs[key]) for key in weight_diffs.keys()]

    plt.figure(figsize=(10, 8))
    plt.bar(range(len(norms)), norms, tick_label=list(weight_diffs.keys()))
    plt.xticks(rotation=90)
    plt.ylabel('Frobenius Norm of Weight Differences')
    plt.title('Comparison of Weight Changes in BERT Layers')
    plt.show() 
    
    
    
def evolution_specific_layer_weight (chekpoint_dir) :
    checkpoint_directory = chekpoint_dir
    checkpoint_files = os.listdir(checkpoint_directory)

    checkpoint_files.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    print("file list",checkpoint_files)
    # Liste pour stocker les normes des poids
    bias_norms = []
   
    # Chargement de chaque checkpoint et calcul de la norme de cls.predictions.bias
    for checkpoint in checkpoint_files:
        model = AutoModelForMaskedLM.from_pretrained(chekpoint_dir + checkpoint)
        bias_weight = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        norm = np.linalg.norm(bias_weight)
        bias_norms.append(norm)

    # Affichage des normes
    epochs = list(range(1, len(bias_norms) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, bias_norms, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm of bert.embeddings.word_embeddings.weight)')
    plt.title('Evolution of L2 Norm of bert.embeddings.word_embeddings.weight Across Epochs')
    plt.grid(True)
    plt.show()
    
    
def drift_layer_weight(baseline,checkpoint_dir):
    
    checkpoint_directory = checkpoint_dir
    checkpoint_files = os.listdir(checkpoint_directory)

    checkpoint_files.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    print("file list",checkpoint_files)
    # Liste pour stocker les normes des poids
    drift = []
    bias_weight=baseline.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    # Chargement de chaque checkpoint et calcul de la norme de cls.predictions.bias
    for checkpoint in checkpoint_files:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir + checkpoint)
        ref_weight = bias_weight
        bias_weight = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        drift.append(np.linalg.norm(bias_weight - ref_weight))

    # Affichage des normes
    epochs = list(range(1, len(drift) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, drift, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm of bert.embeddings.word_embeddings.weight)')
    plt.title('Evolution of L2 Norm of bert.embeddings.word_embeddings.weight Across Epochs')
    plt.grid(True)
    plt.show()
    
    
def evaluate_model_token(model,dataloader):
    model.eval()
    correct_pred=[]
    incorrect_pred=[]

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()} 
        with torch.no_grad():
            outputs=model(**batch)
        predictions=torch.argmax(outputs.logits,dim=-1)
        indices_tokens_masked = torch.nonzero(batch["labels"] != -100, as_tuple=False)
        correct_indices=[]
        incorrect_indices=[]
        for id,label in enumerate(batch["labels"][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]]):
            if label.item() == predictions[indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]][id]:
                correct_indices.append(id)
            else :
                incorrect_indices.append(id)
        correct_pred.extend(batch['labels'][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]][correct_indices])
        incorrect_pred.extend(batch['labels'][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]][incorrect_indices])

    return correct_pred,incorrect_pred


def decoding_text(list,tokenizer):
    counter = Counter(list)

    # Trier les éléments par leur fréquence décroissante
    sorted_numbers = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Extraire les chiffres triés
    unique_sorted_numbers = [num for num, _ in sorted_numbers]

    decoded_texts = []
    for tensor in unique_sorted_numbers:
        decoded_text = tokenizer.decode(tensor.item())
        decoded_texts.append(decoded_text)
    return decoded_texts


def unique_token_ids(dataset):
    unique_token_ids = set()
    for example in dataset:
        unique_token_ids.update(example['input_ids'])
    return torch.tensor(sorted(unique_token_ids))



def change_embedding_word(model1,model2,dataset,tokenizer):
    unique_token_id = unique_token_ids(dataset)
    ref_weight = model1.bert.embeddings.word_embeddings.weight.detach()[unique_token_id]
    bias_weight = model2.bert.embeddings.word_embeddings.weight.detach()[unique_token_id]
    embedding_changes = torch.norm(bias_weight - ref_weight, dim=1)
    values, top_indices = torch.topk(embedding_changes, 5)

    top_words = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
    print(values)

    print("Top 5 words with the most changed embeddings:")
    for word in top_words:
        print(word)
    
     
    
