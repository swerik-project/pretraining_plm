from transformers import Trainer
from transformers import TrainingArguments
import math
import torch


def evaluation_task(model,dataloader):

    print("Trainer evaluation....")
    logging_steps = 500

    training_args = TrainingArguments(
        per_device_eval_batch_size=64,
        logging_steps=logging_steps,)


    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataloader.dataset
    )
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


    print("Manual perplexity...")
    model.eval()
    losses=[]
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(dataloader.batch_size))
        total_loss_eval +=loss.item()


    losses = torch.cat(losses)
    losses = losses[: len(dataloader.dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
       perplexity = float("inf")
    print(f" Perplexity: {perplexity}")


    print("Accuracy...")
    for _, batch in enumerate(dataloader):
        indices_tokens_masked = torch.nonzero(batch["labels"] != -100, as_tuple=False)
        output = model(**batch)
        predicted_token_ids = torch.argmax(output.logits, dim=-1)
        correct_predictions += torch.sum(
            torch.eq(batch["input_ids"][indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]], 
                    predicted_token_ids[indices_tokens_masked[:, 0], indices_tokens_masked[:, 1]])
        ).item()
        total_predictions += indices_tokens_masked.size(0)
        
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)
    
