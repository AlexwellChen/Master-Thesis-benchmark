import torch
import datasets
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, TrainingArguments, AutoConfig
from trainer_accelerate_epoch_val_no_profile import AcceleratorTrainer
from accelerate import Accelerator
import argparse
from adan import Adan
import transformers
import sys
sys.path.append('../')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pyJoules.energy_meter import EnergyMeter
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain

from ls_module.ls_hf_transformer_layer import LSBertForSequenceClassification
from ls_module.hf_args import ModelArguments

from accelerate import DistributedDataParallelKwargs
import random
import numpy as np
import pandas as pd

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def data_process(args):
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_loader, test_loader, eval_loader

def model_and_trainer(train_loader, test_loader, eval_loader, args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.fp16)
    train_args = TrainingArguments(output_dir='benchmark/lightseq_output')
    train_args.fp16 = True if accelerator.mixed_precision == 'fp16' else False
    train_args.local_rank = accelerator.process_index
    config = AutoConfig.from_pretrained('bert-base-cased', num_labels=2)
    model_args = ModelArguments(model_name_or_path='bert-base-cased')
    model_args.module_type = args.module_type
    model = LSBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        training_args=train_args,
        model_args=model_args,
        ls_max_batch_tokens=args.max_batch_tokens,
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
    parser.add_argument('--n_epochs', type=int, default=3)
    # Add the argument for optimizer
    parser.add_argument('--optimizer', type=str, default='adamw')
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
    parser.add_argument('--warmup', type=int, default=50)
    # num_workers
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--module_type', type=int, default=0) # 0 for hugging face, 1 for lightseq
    parser.add_argument('--fp16', type=str, default='fp16')
    # Add the argument for device
    parser.add_argument('--device', type=str, default='v100')
    parser.add_argument('--lightseq', type=str, default='lightseq')

    args = parser.parse_args()

    args.fused_optimizer = True if args.fused_optimizer == 'True' else False
    args.foreach = True if args.foreach == 'True' else False

    if args.batch_size == 8:
        args.max_batch_tokens = 4096
    elif args.batch_size == 16:
        args.max_batch_tokens = 8192
    elif args.batch_size == 32:
        args.max_batch_tokens = 16384
    elif args.batch_size == 64:
        args.max_batch_tokens = 32768

    transformers.set_seed(args.seed)
    seed_torch(args.seed)

    # create df
    df = pd.DataFrame(columns=['optimizer', 'mixed_precision', 'module', 'batch_size', 'device', 'time', 'energy', 'accuracy'])
    
    if args.optimizer == 'adamw':
        args.lr = 5e-5
        args.fused_optimizer = False
    else:
        args.lr = 1e-4
        args.fused_optimizer = True
    args.module_type = 1 if args.lightseq == 'lightseq' else 0
    name = args.optimizer + '_' + args.fp16 + '_' + args.lightseq + '_' + str(args.batch_size)
    
    # Get data
    train_loader, test_loader, eval_loader = data_process(args)
    trainer = model_and_trainer(train_loader, test_loader, eval_loader, args)
    # Init energy meter, add CPU, RAM and GPU
    # Get GPU number
    gpu_num = trainer.accelerator.num_processes
    domains = []
    for i in range(gpu_num):
        domains.append(NvidiaGPUDomain(i))
    device_to_measure = DeviceFactory.create_devices(domains=domains)
    meter = EnergyMeter(device_to_measure)

    # Train the model for n epochs
    meter.start()
    trainer.train(args.n_epochs)
    meter.stop()

    # Save energy trace
    trace = meter.get_trace()
    energy = trace._samples[0].energy['nvidia_gpu_0']

    test_acc = trainer.test()
    
    # append to csv
    # column name: optimizer, mixed_precision, module, batch_size, time, energy, test accuracy
    new_row = {'optimizer': args.optimizer, 'mixed_precision': args.fp16, 'module': args.lightseq, 'batch_size': args.batch_size, 'device': args.device, 'time': "{:.2f}".format(trainer.train_time), 'energy': energy, 'accuracy': test_acc}
    df.loc[len(df)] = new_row
    
    print("Name: ", name)
    print("Energy: ", energy, "mJ")
    print("Total time: ", "{:.2f}".format(trainer.train_time), "s")
    print("Test accuracy: ", test_acc)
    print("=========================================")

                    
    # overwrite csv
    df.to_csv('profiling_'+args.device+"_multiple_times"+'.csv', index=False)
        

    

    
