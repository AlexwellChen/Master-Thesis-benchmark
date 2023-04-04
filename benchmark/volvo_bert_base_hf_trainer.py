import torch
import datasets
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, TrainingArguments, Trainer, AutoConfig
import argparse
from adan import Adan
import transformers
from load_volvo_dataset import load_volvo_dataset_config
import sys
sys.path.append('../')
import time

from pyJoules.energy_meter import EnergyMeter
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain

from ls_module.ls_hf_transformer_layer import LSBertForSequenceClassification
from ls_module.hf_args import ModelArguments

class TimeCallback:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.training_time = None

    def on_epoch_begin(self, trainer, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, trainer, **kwargs):
        self.end_time = time.time()
        epoch_time = self.end_time - self.start_time
        self.training_time += epoch_time
        print(f"Epoch took {epoch_time:.2f} seconds")

def model_and_trainer(train_loader, test_loader, eval_loader, args, config):
    
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    train_args = TrainingArguments( output_dir='benchmark/lightseq_output', 
                                    num_train_epochs=args.n_epochs, 
                                    fp16=args.fp16,
                                    per_device_train_batch_size=args.batch_size,
                                    per_device_eval_batch_size=args.batch_size*4,
                                    metric_for_best_model='accuracy',
                                   )

    model_args = ModelArguments(model_name_or_path='bert-base-cased')
    model_args.module_type = args.module_type
    # print(config)
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
        betas = (0.98, 0.92, 0.9)
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
    
    # model, optimizer, train_loader, scheduler = accelerator.prepare(
    #         model, optimizer, train_loader, scheduler
    #     )
    time_callback = TimeCallback()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=[optimizer, scheduler],
        callbacks=[time_callback]
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
    # num_workers
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--module_type', type=int, default=0) # 0 for hugging face, 1 for lightseq
    parser.add_argument('--fp16', type=bool, default=False)
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

    train_loader, test_loader, eval_loader, config = load_volvo_dataset_config(args)
    trainer = model_and_trainer(train_loader, test_loader, eval_loader, args, config)

    # Init energy meter, add CPU, RAM and GPU
    # Get GPU number
    gpu_num = 1
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

    test_results = trainer.evaluate(test_loader)

    # print total energy consumption in xx.xx kJ format
    # print("Total energy consumption: ", "{:.2f}".format(trainer.total_energy), "kJ")
    # print total time in xx.xx s format
    print("Total time: ", "{:.2f}".format(trainer.train_time), "s")
    print("Test res: ", test_results)
    # write avg sm occupancy, time in ./benchmark/metrics/log_file_name.txt
    with open('./benchmark/metrics/'+args.log_file_name+'.txt', 'w') as f:
        f.write("Total time: ")
        f.write("{:.2f}".format(trainer.train_time))
        f.write('\n')
        f.write("Test res: ")
        f.write("{:.2f}".format(test_results))
        

    # # save loss values in ./loss_val/ folder
    # loss = trainer.log_history['loss']
    # # save original loss values in ./loss_val/ folder
    # with open('./benchmark/loss_val/'+args.log_file_name+'_loss.txt', 'w') as f:
    #     for item in loss:
    #         f.write(str(item))
    #         f.write('\n')

    # # save accuracy values in ./acc_val/ folder
    # accuracy = [item['accuracy'] for item in trainer.val_logs]
    # # save original accuracy values in ./acc_val/ folder
    # with open('./benchmark/acc_val/'+args.log_file_name+'_acc.txt', 'w') as f:
    #     for item in accuracy:
    #         f.write(str(item))
    #         f.write('\n')

    # # save log_history in ./log_history/ folder
    # with open('./benchmark/log_history/'+args.log_file_name+'_log.txt', 'w') as f:
    #     for item in trainer.log_history:
    #         f.write(str(item))
    #         f.write('\n')
    