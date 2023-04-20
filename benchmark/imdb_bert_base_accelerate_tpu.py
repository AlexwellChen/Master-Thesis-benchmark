import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
from transformers.file_utils import is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
print(is_torch_tpu_available())

# imports pytorch
import torch
# imports the adamw optimizer
from transformers import AdamW

# imports the torch_xla package

import datasets
from accelerate import Accelerator

hyperparameters = {
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "logging_dir": "./tpu_logs",
    "logging_steps": 100,
    "save_steps": 500,
    "save_total_limit": 2,
    "seed": 42,
}

def data_process():
    # Define the function to encode the data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')
    
    # Load the IMDB dataset and create data loaders for training, validation and test
    train_dataset, test_dataset = datasets.load_dataset('imdb', split=['train', 'test'])
    # reduce test dataset size to original size*0.2
    test_dataset = test_dataset.select(range(int(len(test_dataset)*0.2)))
    split_set = train_dataset.train_test_split(test_size=0.1, seed=38)
    train_dataset = split_set['train']
    eval_dataset = split_set['test']

    
    train_dataset = train_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters["train_batch_size"], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters["eval_batch_size"], shuffle=False)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=hyperparameters["eval_batch_size"], shuffle=False)

    return train_dataloader, test_dataloader, eval_dataloader


def training_function(train_dataloader, test_dataloader, eval_dataloader):
    # Initialize accelerator
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    # Now we train the model
    import tqdm as tqdm
    progess_bar = tqdm.tqdm(range(num_epochs*len(train_dataloader)), desc="Training")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            progess_bar.update(1)

            if step % 100 == 0:
                print(f"Epoch {epoch} - Step {step} - Loss {loss}")
        # We evaluate the model at the end of each epoch
        model.eval()
        eval_accuracy = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                accuracy = outputs.accuracy
            eval_accuracy += accuracy
        eval_accuracy /= len(eval_dataloader)
        print(f"Epoch {epoch} - validation Loss {loss} - Accuracy {accuracy}")
    
    # test the model
    model.eval()
    eval_accuracy = 0
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            accuracy = outputs.accuracy
        eval_accuracy += accuracy
    eval_accuracy /= len(test_dataloader)
    print(f"Test Loss {loss} - Accuracy {accuracy}")

if __name__ == "__main__":
    # dataset
    train_dataloader, test_dataloader, eval_dataloader = data_process()
    training_function(train_dataloader, test_dataloader, eval_dataloader)