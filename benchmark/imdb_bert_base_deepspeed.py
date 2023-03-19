import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import argparse
from adan import Adan
import transformers
import torch.distributed as dist

from pyJoules.energy_meter import EnergyMeter
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

import os
from datasets import load_metric
from transformers import DataCollatorWithPadding
import numpy as np


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
    

    return train_dataset, test_dataset, eval_dataset, tokenizer

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}


def model_and_trainer(train_dataset, test_dataset, eval_dataset, tokenizer, args):
    # Load the pre-trained "bert-base-cased" model and add a linear layer on top for classification
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
    model = PipelineModule(layers=model,
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method='parameters',
                         activation_checkpoint_interval=0)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    train_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=args.n_epochs,              # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
        weight_decay=args.wd,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=50,
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,
        # deepspeed
        deepspeed='./deepspeed_config.json',
    )

    # Instantiate the Trainer class and pass in the required parameters
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
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
    # Wether to use fused optimizer
    parser.add_argument('--fused_optimizer', type=str, default='False')
    # Wether to use foreach
    parser.add_argument('--foreach', type=str, default='True')
    # Weight decay
    parser.add_argument('--wd', type=float, default=0.01)
    # Warmup steps
    parser.add_argument('--warmup', type=int, default=320)
    # Seed
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
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

    # Init energy meter, add CPU, RAM and GPU
    gpu_num = torch.cuda.device_count()
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
    handler = CSVHandler("./benchmark/energy/"+args.log_file_name + '_Energy_Results.csv')
    handler.process(trace)
    handler.save_data()

    test_acc = trainer.test()
    
    # print avg sm occupancy in xx.xx% format
    print("Avg SM occupancy: ", "{:.2f}".format(trainer.avg_sm_occupancy), "%")
    # print total energy consumption in xx.xx kJ format
    # print("Total energy consumption: ", "{:.2f}".format(trainer.total_energy), "kJ")
    # print total time in xx.xx s format
    print("Total time: ", "{:.2f}".format(trainer.train_time), "s")
    print("Test accuracy: ", "{:.2f}".format(test_acc), "%")
    # write avg sm occupancy, time in ./benchmark/metrics/log_file_name.txt
    with open('./benchmark/metrics/'+args.log_file_name+'.txt', 'w') as f:
        f.write("Avg SM occupancy: ")
        f.write("{:.2f}".format(trainer.avg_sm_occupancy))
        f.write("%")
        f.write('\n')
        f.write("Total time: ")
        f.write("{:.2f}".format(trainer.train_time))
        f.write('\n')
        f.write("Test accuracy: ")
        f.write("{:.2f}".format(test_acc))

        

    # save loss values in ./loss_val/ folder
    loss = [item['loss'] for item in trainer.training_logs]
    # save original loss values in ./loss_val/ folder
    with open('./benchmark/loss_val/'+args.log_file_name+'_loss.txt', 'w') as f:
        for item in loss:
            f.write(str(item))
            f.write('\n')

    # save accuracy values in ./acc_val/ folder
    accuracy = [item['accuracy'] for item in trainer.val_logs]
    # save original accuracy values in ./acc_val/ folder
    with open('./benchmark/acc_val/'+args.log_file_name+'_acc.txt', 'w') as f:
        for item in accuracy:
            f.write(str(item))
            f.write('\n')

    
