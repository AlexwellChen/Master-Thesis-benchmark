import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from trainer import ProfilingTrainer
import argparse
from adan import Adan


def data_process(args):
    # Define the function to encode the data
    # Load the CoLa dataset and create data loaders for training and validation
    train_dataset, eval_dataset = datasets.load_dataset('glue', 'cola', split=['train', 'validation'])
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    train_dataset = train_dataset.map(lambda data: tokenizer(data["sentence"],
                                                        padding="max_length",
                                                        truncation=True,
                                                        max_length=512),
                                                        batched=True,
                                                        batch_size=args.batch_size,
                                                        drop_last_batch=False)
    eval_dataset = eval_dataset.map(lambda data: tokenizer(data["sentence"],
                                                        padding="max_length",
                                                        truncation=True,
                                                        max_length=512),
                                                        batched=True,
                                                        batch_size=args.batch_size,
                                                        drop_last_batch=False)

    train_dataset.set_format(type='torch', columns=['label', 'idx', 'input_ids', 'attention_mask'])
    eval_dataset.set_format(type='torch', columns=['label', 'idx', 'input_ids', 'attention_mask'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_loader, eval_loader

def model_and_trainer(train_loader, eval_loader, args):
    # Load the pre-trained "roberta-base" model and add a linear layer on top for classification
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2, dropout=0.1, attention_dropout=0.1)

    # Define the optimizer and learning rate scheduler
    if args.optimizer == 'adam':
        betas = (0.9,0.98)
        if args.fused_optimizer and args.foreach:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1, fused=True, foreach=True, betas=betas, eps=1e-6)
        elif args.fused_optimizer and not args.foreach:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1, fused=True, foreach=False, betas=betas, eps=1e-6)
        elif not args.fused_optimizer and args.foreach:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1, fused=False, foreach=True, betas=betas, eps=1e-6)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1, fused=False, foreach=False, betas=betas, eps=1e-6)
    elif args.optimizer == 'adan':
        betas = (0.98, 0.99, 0.99)
        if args.fused_optimizer and args.foreach:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True, foreach=True, betas=betas, eps=1e-8)
        elif args.fused_optimizer and not args.foreach:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True, foreach=False, betas=betas, eps=1e-8)
        elif not args.fused_optimizer and args.foreach:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=0.01, fused=False, foreach=True, betas=betas, eps=1e-8)
        else:
            optimizer = Adan(model.parameters(), lr=args.lr, weight_decay=0.01, fused=False, foreach=False, betas=betas, eps=1e-8)
    
    # polynomial_decay, warmup_updates: 320
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
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
    parser.add_argument('--n_epochs', type=int, default=10)
    # Add the argument for optimizer
    parser.add_argument('--optimizer', type=str, default='adam')
    # Add the argument for learning rate
    parser.add_argument('--lr', type=float, default=1e-5) # Adam: 1e-5, Adan: 4e-5
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
    
