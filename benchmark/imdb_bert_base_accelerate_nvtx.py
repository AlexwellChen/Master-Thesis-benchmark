import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, TrainingArguments, AutoConfig
from trainer_accelerate_nvtx import AcceleratorTrainer
from accelerate import Accelerator
import argparse
from adan import Adan
import transformers
import sys
sys.path.append('../')

from pyJoules.energy_meter import EnergyMeter
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain

from ls_module.ls_hf_transformer_layer import LSBertForSequenceClassification
from ls_module.hf_args import ModelArguments


def data_process(args):
    # Define the function to encode the data
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')
    
    # Load the IMDB dataset and create data loaders for training, validation and test
    train_dataset, test_dataset = datasets.load_dataset('imdb', split=['train', 'test'])
    # reduce test dataset size to original size*0.2
    test_dataset = test_dataset.select(range(int(len(test_dataset)*0.2)))
    split_set = train_dataset.train_test_split(test_size=0.1, seed=38)
    train_dataset = split_set['train']
    eval_dataset = split_set['test']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_dataset = train_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_loader, test_loader, eval_loader

def model_and_trainer(train_loader, test_loader, eval_loader, args):
    accelerator = Accelerator()
    train_args = TrainingArguments(output_dir='benchmark/lightseq_output')
    train_args.fp16 = True if accelerator.mixed_precision == 'fp16' else False
    train_args.local_rank = accelerator.process_index
    config = AutoConfig.from_pretrained('bert-base-cased', num_labels=2)
    model_args = ModelArguments(model_name_or_path='bert-base-cased')
    model_args.module_type = args.module_type
    print(config)
    model = LSBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        training_args=train_args,
        model_args=model_args,
        config=config,
    )
        
    # Define the optimizer and learning rate scheduler
    if args.optimizer == 'adam':
        # betas = (0.9, 0.999) #default
        if args.foreach:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=True, eps=1e-8)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False, eps=1e-8)
    elif args.optimizer == 'adan':
        betas = (0.98, 0.92, 0.99)
        if args.fused_optimizer and args.foreach:
            optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True, foreach=True, betas=betas, eps=1e-8)
        elif args.fused_optimizer and not args.foreach:
            optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True, foreach=False, betas=betas, eps=1e-8)
        elif not args.fused_optimizer and args.foreach:
            optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.wd, fused=False, foreach=True, betas=betas, eps=1e-8)
        else:
            optimizer = Adan(params=model.parameters(), lr=args.lr, weight_decay=args.wd, fused=False, foreach=False, betas=betas, eps=1e-8)
    # adamw
    elif args.optimizer == 'adamw':
        betas=(0.9, 0.999)
        if args.fused_optimizer:
            # runtime error: Not supported: FusedAdamW
            print('Not supported: Fused AdamW')
        if args.foreach:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=True, betas=betas, eps=1e-8)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False, betas=betas, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup, 
                                                num_training_steps=len(train_loader) * args.n_epochs
                                            )
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    # Instantiate the ProfilingTrainer class and pass in the required parameters
    trainer = AcceleratorTrainer(
        model=model,
        accelerator=accelerator,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        test_dataloader=test_loader,
        optimizers=[optimizer, scheduler],
        device=accelerator.device,
        n_steps_per_val=args.n_steps_per_val,
        target_val_acc=args.target_val_acc,
        log_file_name=args.log_file_name,
        seed=args.seed
    )

    return trainer

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=5)
    # Add the argument for optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    # Add the argument for learning rate
    parser.add_argument('--lr', type=float, default=5e-5)
    # Add the argument for batch size
    parser.add_argument('--batch_size', type=int, default=16)
    # Add the argument for number of n_steps_per_val
    parser.add_argument('--n_steps_per_val', type=int, default=50)
    # Add the argument for target_val_acc
    parser.add_argument('--target_val_acc', type=str, default=None)
    # Add the name for log file
    parser.add_argument('--log_file_name', type=str, default='profiling')
    # Whether to use fused optimizer
    parser.add_argument('--fused_optimizer', type=str, default='False')
    # Whether to use foreach
    parser.add_argument('--foreach', type=str, default='True')
    # Weight decay
    parser.add_argument('--wd', type=float, default=0.01)
    # Warmup steps
    parser.add_argument('--warmup', type=int, default=320)
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--module_type', type=int, default=0) # 0 for hugging face, 1 for lightseq
    args = parser.parse_args()

    args.fused_optimizer = True if args.fused_optimizer == 'True' else False
    args.foreach = True if args.foreach == 'True' else False

    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    if args.target_val_acc is None:
        print('No target_val_acc specified')
    else:
        args.target_val_acc = float(args.target_val_acc)
        print('target_val_acc: ', args.target_val_acc)

    train_loader, test_loader, eval_loader = data_process(args)
    trainer = model_and_trainer(train_loader, test_loader, eval_loader, args)

    # Train the model for n epochs
    torch.cuda.cudart().cudaProfilerStart()
    trainer.train(args.n_epochs)
    torch.cuda.cudart().cudaProfilerStop()
