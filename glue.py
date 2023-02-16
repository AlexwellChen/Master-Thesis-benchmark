import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from trainer import ProfilingTrainer

# Load the MRPC dataset and create data loaders for training and validation
train_dataset, eval_dataset = datasets.load_dataset('glue', 'mrpc', split=['train', 'validation'])
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
train_encodings = tokenizer(train_dataset['sentence1'], train_dataset['sentence2'], truncation=True, padding=True)
eval_encodings = tokenizer(eval_dataset['sentence1'], eval_dataset['sentence2'], truncation=True, padding=True)
train_dataset = datasets.Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_dataset['label']})
eval_dataset = datasets.Dataset.from_dict({'input_ids': eval_encodings['input_ids'], 'attention_mask': eval_encodings['attention_mask'], 'labels': eval_dataset['label']})
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)

# Load the pre-trained "bert-base-cased" model and add a linear layer on top for classification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=len(train_loader) * 3
                                        )

# Instantiate the ProfilingTrainer class and pass in the required parameters
trainer = ProfilingTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=eval_loader,
    optimizers=[optimizer, scheduler],
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    n_steps_per_val=100,
    target_val_acc=0.85
)

# Train the model for 3 epochs
trainer.train(3)

# Print the profiling information
print(trainer.profiler_log)
