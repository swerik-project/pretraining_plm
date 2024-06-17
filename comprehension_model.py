import matplotlib.pyplot as plt
import numpy as np
import os
import re
from transformers import AutoModelForMaskedLM
import torch
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    
    
def plot_weight_distributions(model1, model2, layer_name):
    weights1 = model1.state_dict()[layer_name].flatten().cpu().numpy()
    weights2 = model2.state_dict()[layer_name].flatten().cpu().numpy()
    print(weights1)

    plt.figure(figsize=(10, 5))
    plt.hist(weights1, bins=100, alpha=0.5, label='Fine tuned Model')
    plt.hist(weights2, bins=100, alpha=0.5, label='Baseline Model')
    plt.title(f"Weight Distribution Comparison for {layer_name}")
    plt.xlabel("Weight values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_hidden_states_distributions(model1, model2, dataloader, tokenizer):
    hidden_states1 = get_embeddings(model1, dataloader, tokenizer)
    hidden_states2 = get_embeddings(model2, dataloader, tokenizer)

    for i in range(len(hidden_states1)):
        plt.figure(figsize=(10, 5))
        plt.hist(hidden_states1[i].flatten(), bins=100, alpha=0.5, label='Baseline Model')
        plt.hist(hidden_states2[i].flatten(), bins=100, alpha=0.5, label='Fine-tuned Model')
        plt.title(f"Hidden States Distribution Comparison for Layer {i}")
        plt.xlabel("Hidden States Values")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    
def get_embeddings(model, dataloader, tokenizer):
    model.eval()
    model.to(device)
    layerwise_embeddings = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for batch in dataloader:
        if batch['input_ids'].numel() == 0:
            continue
        tokens={key :value.to(device).long() for key,value in batch.items()}
        outputs= model(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"],labels=tokens["labels"],output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (layer+1) tensors, each of shape (batch_size, seq_len, hidden_size)
        for i, hidden_state in enumerate(hidden_states):
            cls_embeddings = hidden_state[:, 0, :].detach().cpu().numpy()  # Extract [CLS] token
            layerwise_embeddings[i].append(cls_embeddings)
    return [np.vstack(layer) for layer in layerwise_embeddings]
 
    
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
    
    
# Function to perform PCA and plot for each layer
def plot_pca_for_layers(baseline_embeddings, finetuned_embeddings):
    num_layers = len(baseline_embeddings)
    for layer in range(num_layers):
        combined_embeddings = np.concatenate([baseline_embeddings[layer], finetuned_embeddings[layer]])
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(combined_embeddings)
        num_baseline = baseline_embeddings[layer].shape[0]
        
        plt.figure(figsize=(10, 5))
        plt.scatter(reduced_embeddings[:num_baseline, 0], reduced_embeddings[:num_baseline, 1], label='Baseline', color='blue')
        plt.scatter(reduced_embeddings[num_baseline:, 0], reduced_embeddings[num_baseline:, 1], label='Fine-tuned', color='red')
        plt.legend()
        if layer == 0 :
            plt.title(f"PCA of Word Embeddings at inital state")
        else : 
            plt.title(f"PCA of Word Embeddings at Layer {layer}")
        plt.show()
    
    
# Function to perform PCA and plot for each layer
def plot_tSNE_for_layers(baseline_embeddings, finetuned_embeddings):
    num_layers = len(baseline_embeddings)
    for layer in range(num_layers):
        combined_embeddings = np.concatenate([baseline_embeddings[layer], finetuned_embeddings[layer]])
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)
        num_baseline = baseline_embeddings[layer].shape[0]
        plt.figure(figsize=(10, 5))
        plt.scatter(reduced_embeddings[:num_baseline, 0], reduced_embeddings[:num_baseline, 1], label='Baseline', color='blue')
        plt.scatter(reduced_embeddings[num_baseline:, 0], reduced_embeddings[num_baseline:, 1], label='Fine-tuned', color='red')
        plt.legend()
        if layer == 0 :
            plt.title(f"t-SNE of Word Embeddings at inital state")
        else : 
            plt.title(f"t-SNE of Word Embeddings at Layer {layer}")
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



def change_embedding_word(model1,model2,dataset,tokenizer,number):
    unique_token_id = unique_token_ids(dataset)
    ref_weight = model1.bert.embeddings.word_embeddings.weight.detach()[unique_token_id]
    bias_weight = model2.bert.embeddings.word_embeddings.weight.detach()[unique_token_id]
    embedding_changes = torch.norm(bias_weight - ref_weight, dim=1)
    values, top_indices = torch.topk(embedding_changes, number)

    top_words = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices]
    print(values)

    print(f"Top {number} words with the most changed embeddings:")
    for word in top_words:
        print(word)
    return top_words,embedding_changes
    
    
def transfer_attention_weights(source_model, target_model, layer_index,head_index):
    # Obtenir les poids de la tête d'attention source
    # target_model.bert.encoder.layer[layer_index].attention=source_model.bert.encoder.layer[layer_index].attention
    source_attention = source_model.bert.encoder.layer[layer_index].attention.self
    target_attention = target_model.bert.encoder.layer[layer_index].attention.self
    # target_output = target_model.bert.encoder.layer[layer_index].attention.output
    # source_output = source_model.bert.encoder.layer[layer_index].attention.output
    # Transférer les poids
    start_index = 64*head_index
    end_index= 64*head_index
    # Transférer les poids de manière sûre
    with torch.no_grad():
        target_attention.query.weight.data[:, start_index:end_index] = source_attention.query.weight.data[:, start_index:end_index].clone().detach()
        target_attention.key.weight.data[:, start_index:end_index] = source_attention.key.weight.data[:, start_index:end_index].clone().detach()
        target_attention.value.weight.data[:, start_index:end_index] = source_attention.value.weight.data[:, start_index:end_index].clone().detach()


def transfer_embedding_weights(source_model,target_model):
    source_embeddings = source_model.bert.embeddings
    target_embedding = target_model.bert.embeddings
    
    target_embedding.word_embeddings.weight= source_embeddings.word_embeddings.weight
    target_embedding.position_embeddings.weight= source_embeddings.position_embeddings.weight
    target_embedding.token_type_embeddings.weight= source_embeddings.token_type_embeddings.weight
    target_embedding.LayerNorm.weight= source_embeddings.LayerNorm.weight
    target_embedding.LayerNorm.bias= source_embeddings.LayerNorm.bias
    
def transfer_head_layer(source_model,target_model):
    
    # source_predictions = source_model.cls.predictions.transform
    # target_predictions = target_model.cls.predictions.transform
    target_model.cls.predictions =source_model.cls.predictions
    # target_predictions.dense.weight= source_predictions .dense.weight
    # target_predictions.dense.bias= source_predictions .dense.bias
    # target_predictions.LayerNorm.weight= source_predictions .LayerNorm.weight
    # target_predictions.LayerNorm.bias= source_predictions .LayerNorm.bias
  
def transfer_feed_forward(source_model,target_model,layer_index):
    # source_predictions = source_model.bert.encoder.layer[layer_index].intermediate
    #target_model.bert.encoder.layer[layer_index].intermediate = source_model.bert.encoder.layer[layer_index].intermediate
    # target_predictions = target_model.bert.encoder.layer[layer_index].intermediate
    # source_output = source_model.bert.encoder.layer[layer_index].output
   target_model.bert.encoder.layer[layer_index].output = source_model.bert.encoder.layer[layer_index].output
    # target_output = target_model.bert.encoder.layer[layer_index].output
    
    # target_predictions.dense.weight= source_predictions.dense.weight
    # target_predictions.dense.bias= source_predictions.dense.bias
    # target_output.dense.weight= source_output.dense.weight
    # target_output.dense.bias= source_output.dense.bias
    # target_output.LayerNorm.weight= source_output.LayerNorm.weight
    # target_output.LayerNorm.bias= source_output.LayerNorm.bias

     
    
def extract_activations(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    return hidden_states

def extract_and_compare_activations(model_pre, model_post, dataloader,token_id):
    similarities_hidden_states = []
    similarities_attention=[]
    diff_tot=[]
    
    for batch in dataloader:
        batch = {k: batch[k].to(device) for k in batch.keys()}
        hidden_states_pre = []
        hidden_states_post= []
        with torch.no_grad():
            pre_output= model_pre(**batch, output_hidden_states=True, output_attentions=True)
            pre_activations = pre_output.hidden_states
            pre_attention = pre_output.attentions
            post_output = model_post(**batch, output_hidden_states=True,output_attentions=True)
            post_activations = post_output.hidden_states
            post_attention = post_output.attentions
            #pre_logits= pre_output[1]
            #post_logits = post_output[1]
        masked_indices = (batch['labels'] == token_id).nonzero(as_tuple=True)
        for i in range(len(pre_activations)):
            hidden_states_pre.append(pre_activations[i][masked_indices])
            hidden_states_post.append(post_activations[i][masked_indices])
            
        example_similarities_hidden_states = [cosine_similarity(hidden_states_pre[layer], hidden_states_post[layer]) for layer in range(len(pre_activations))]
        # example_similarities_hidden_states.append(cosine_similarity(pre_logits,post_logits))
        example_similarities_attention = [cosine_similarity(pre_attention[layer], post_attention[layer]) for layer in range(len(pre_attention))]
        diff = [np.linalg.norm((hidden_states_pre[layer] - hidden_states_post[layer]).cpu().numpy()) for layer in range(len(pre_attention))]
        del pre_output
        del post_output
        del pre_attention
        del post_attention
       # del pre_logits
       # del post_logits
        similarities_hidden_states.append(example_similarities_hidden_states)
        similarities_attention.append(example_similarities_attention)
        diff_tot.append(diff)
    
    mean_similarities_hidden_states = np.mean(similarities_hidden_states, axis=0)
    mean_similarities_attention = np.mean(similarities_attention, axis=0)
    mean_diff=np.mean(diff_tot,axis=0)
    return mean_similarities_hidden_states,mean_similarities_attention,mean_diff

def cosine_similarity(tensor1, tensor2):
    # Ensure tensors are flattened (1D) to compute vector cosine similarity
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    cos_sim = torch.nn.functional.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0))
    return cos_sim.item()

def compare_activations(pre_ft_activations, post_ft_activations):
    differences = {}
    for layer in range(len(pre_ft_activations)):
        sim = cosine_similarity(pre_ft_activations[layer], post_ft_activations[layer])
        differences[layer] = sim
    return differences