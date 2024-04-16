from transformers import Trainer
from transformers import TrainingArguments
import math
import torch

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





def evaluation_task(model,dataloader):
    print("Trainer evaluation....")
    eval_results = trainer_evaluation(model,dataloader)
    print(eval_results)
    loss=next(iter(eval_results))
    print(f">>> Perplexity: {math.exp(eval_results[loss]):.2f}")

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
        
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)
    
