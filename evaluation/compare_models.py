from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
import Levenshtein
import argparse
import logging
import torch
import os
from bidict import bidict
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

LOGGER = logging.getLogger(__name__)

def encode(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ix, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['content'],
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
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


def evaluate(model, loader):
    loss1, accuracy1 = 0.0, []
    model.eval()
    true_labels, pred_labels1 = [], []
    misclassified_indices = []
    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader))):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        output1 = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss1 += output1.loss.item()
        preds_batch1 = torch.argmax(output1.logits, axis=1)
        batch_acc1 = torch.mean((preds_batch1 == labels).float())
        accuracy1.append(batch_acc1)
        pred_labels1.extend(preds_batch1.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        for i, (pred, true) in enumerate(zip(preds_batch1, labels)):
            if pred != true:
                misclassified_indices.append(batch_idx * args.batch_size + i)
        
       
       
    train_loss_avg = loss1 * args.batch_size / len(loader)
    print(f'Training Loss 1: {train_loss_avg:.3f}')
    print("\nAccuracy model 1:", accuracy_score(true_labels, pred_labels1))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels1))
        
    return pred_labels1, true_labels, misclassified_indices
    


def main(args):
    os.makedirs(args.model_filename1, exist_ok=True)
    logging.basicConfig(filename=f'{args.model_filename1}/evaluation.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Create binary label where seg = 1
    df = df[df["content"].notnull()]
    label_names = args.label_names
    print(label_names)
    if label_names is None:
        label_names = sorted(list(set(df["tag"])))
    label_dict = {ix: name for ix, name in enumerate(label_names)}
    df["tag"] = [bidict(label_dict).inv[tag] for tag in df["tag"]]
    print(len(label_names))
    print(label_dict)

    LOGGER.debug("load model...")
    model1 = AutoModelForSequenceClassification.from_pretrained(
        args.model_filename1,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)

    model2 = AutoModelForSequenceClassification.from_pretrained(
        args.model_filename2,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)
    LOGGER.info("Load and tokenizer...")
    tokenizer1 = AutoTokenizer.from_pretrained(args.tokenizer1)
    tokenizer2= AutoTokenizer.from_pretrained(args.tokenizer2)
    LOGGER.info("Preprocess datasets...")
    input_ids1, attention_masks1, labels1 = encode(df, tokenizer1)
    input_ids2, attention_masks2, labels2 = encode(df, tokenizer2)

    LOGGER.info(f"Labels: {labels1}")

    dataset1 = TensorDataset(input_ids1, attention_masks1, labels1)
    dataset2 = TensorDataset(input_ids2, attention_masks2, labels2)
    test_loader1 = DataLoader(
        dataset1,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    test_loader2 = DataLoader(
        dataset2,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    pred1, true_label, misclassified_indices1 = evaluate(model1, test_loader1)
    pred2, true_label, misclassified_indices2 = evaluate(model2, test_loader2)
    
    # Find correctly classified by model2 but not by model1
    correct_by_model2 = set(range(len(df))) - set(misclassified_indices2)
    misclassified_by_model1 = set(misclassified_indices1)
    correct_by_model2_not_by_model1 = correct_by_model2 & misclassified_by_model1
    correct_rows = df.iloc[list(correct_by_model2_not_by_model1)]
    output_file = os.path.join(args.model_filename2, 'correct_by_model2_not_by_model1.csv')
    correct_rows.to_csv(output_file, index=False)

    print("Test McNemar")
    # Calculer le nombre de misclassifications
    misclassified_by_1_not_by_2 = 0
    misclassified_by_2_not_by_1 = 0
    

    # Itérer sur les listes de labels
    for pl1, pl2, tl in zip(pred1, pred2, true_label):
        # Vérifier les misclassifications
        if pl1 != tl and pl2 == tl:
            misclassified_by_1_not_by_2 += 1
        elif pl2 != tl and pl1 == tl:
            misclassified_by_2_not_by_1 += 1
            
    
    table = np.array([[0, misclassified_by_1_not_by_2],
                    [misclassified_by_2_not_by_1, 0]]) # contingency_tables
    
    x = (abs(misclassified_by_1_not_by_2-misclassified_by_2_not_by_1)-1)**2/(misclassified_by_1_not_by_2+misclassified_by_2_not_by_1)

    # test  of McNemar
    result = mcnemar(table, exact=False, correction=True)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    if result.pvalue < 0.05:
        print('Differences between models are significant')
    else:
        print('Differences between models are not significant')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename1", type=str, default="trained/binary_note_seg_model", help="Save location for the model")
    parser.add_argument("--model_filename2", type=str, default="trained/binary_note_seg_model", help="Save location for the model")
    parser.add_argument("--base_model", type=str, default="KBLab/bert-base-swedish-cased", help="Base model that the model is initialized from")
    parser.add_argument("--tokenizer1", type=str, default="KBLab/bert-base-swedish-cased", help="Which tokenizer to use; accepts local and huggingface tokenizers.")
    parser.add_argument("--tokenizer2", type=str, default="KBLab/bert-base-swedish-cased", help="Which tokenizer to use; accepts local and huggingface tokenizers.")
    parser.add_argument("--label_names", type=str, nargs="+", default=None, help="A list of label names to be used in the classifier. If None, takes class names from 'tag' column in the data.")
    parser.add_argument("--data_path", type=str, default="data/pilot/val_processed_data.csv", help="Testing data as a .CSV file. Needs to have 'content' (X) and 'tag' (Y) columns.")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use for training. Use 'cpu' for CPU.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
