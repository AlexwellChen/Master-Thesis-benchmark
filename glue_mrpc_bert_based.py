import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from trainer import ProfilingTrainer
import argparse
from adan import Adan


def data_process(args):
    # Define the function to encode the data
    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
    # Load the MRPC dataset and create data loaders for training and validation
    train_dataset, eval_dataset = datasets.load_dataset('glue', 'mrpc', split=['train', 'validation'])
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    train_dataset = train_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_loader, eval_loader

def model_and_trainer(train_loader, eval_loader, args):
    # Load the pre-trained "bert-base-cased" model and add a linear layer on top for classification
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

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
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=args, fused=True, foreach=True, betas=betas, eps=1e-8)
        elif args.fused_optimizer and not args.foreach:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True, foreach=False, betas=betas, eps=1e-8)
        elif not args.fused_optimizer and args.foreach:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=False, foreach=True, betas=betas, eps=1e-8)
        else:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=False, foreach=False, betas=betas, eps=1e-8)
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

    # Instantiate the ProfilingTrainer class and pass in the required parameters
    trainer = ProfilingTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        optimizers=[optimizer, scheduler],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        n_steps_per_val=args.n_steps_per_val,
        target_val_acc=args.target_val_acc,
        log_file_name=args.log_file_name
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
    parser.add_argument('--target_val_acc', type=float, default=None)
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

    args = parser.parse_args()

    args.fused_optimizer = True if args.fused_optimizer == 'True' else False
    args.foreach = True if args.foreach == 'True' else False

    train_loader, eval_loader = data_process(args)
    trainer = model_and_trainer(train_loader, eval_loader, args)
    # Train the model for 3 epochs
    trainer.train(args.n_epochs)

    # print avg sm occupancy in xx.xx% format
    print("Avg SM occupancy: ", "{:.2f}".format(trainer.avg_sm_occupancy), "%")
    # print total energy consumption in xx.xx kJ format
    # print("Total energy consumption: ", "{:.2f}".format(trainer.total_energy), "kJ")
    # print total time in xx.xx s format
    print("Total time: ", "{:.2f}".format(trainer.train_time), "s")

    # plot the loss curve
    import matplotlib.pyplot as plt
    loss = [item['loss'] for item in trainer.training_logs]
    # save original loss values in ./loss_val/ folder
    with open('./loss_val/'+args.log_file_name+'_loss.txt', 'w') as f:
        for item in loss:
            f.write(str(item))
            f.write('\n')
    smooth_loss = [sum(loss[max(0, i-10):i+1])/len(loss[max(0, i-10):i+1]) for i in range(len(loss))]
    # save the loss curve
    plt.plot(smooth_loss)
    plt.savefig('./loss_fig/'+args.log_file_name+'_loss.png')

    # plot the accuracy curve
    accuracy = [item['accuracy'] for item in trainer.val_logs]
    # save original accuracy values in ./acc_val/ folder
    with open('./acc_val/'+args.log_file_name+'_acc.txt', 'w') as f:
        for item in accuracy:
            f.write(str(item))
            f.write('\n')
    # save the accuracy curve
    plt.plot(accuracy)
    plt.savefig('./acc_fig/'+args.log_file_name+'_acc.png')
    