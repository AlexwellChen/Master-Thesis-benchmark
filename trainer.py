import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import evaluate

class ProfilingTrainer:
    def __init__(
            self, model, train_dataloader, val_dataloader, optimizers, 
            device, n_steps_per_val, target_val_acc, log_file_name='default_log'
        ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizers[0]
        self.scheduler = optimizers[1]
        self.device = device
        self.n_steps_per_val = n_steps_per_val
        self.target_val_acc = target_val_acc
        self.training_logs = []
        self.val_logs = []
        self.train_time = 0
        self.profiler_log = None
        self.log_file_name = log_file_name
        
    def train(self, n_epochs):
        self.model.to(self.device)
        self.optimizer.zero_grad()
        train_start_time = time.time()

        progress_bar = tqdm(range(n_epochs * len(self.train_dataloader)), desc="Epoch")
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+self.log_file_name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        with prof:
            for epoch in range(n_epochs):
                epoch_start_time = time.time()
                self.model.train()
                for step, batch in enumerate(self.train_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.model.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    prof.step()
                    if (step + 1) % self.n_steps_per_val == 0:
                        val_acc = self.evaluate()
                        self.val_logs.append({'step': step, 'accuracy': val_acc})
                        print(f"Validation accuracy at step {step+1}: {val_acc:.2f}, loss: {loss.item():.2f}")
                        if val_acc >= self.target_val_acc:
                            print(f"Stopping training at epoch {epoch+1}, step {step+1} as target validation accuracy reached")
                            self.train_time = time.time() - train_start_time
                            # self.profiler_log = prof.key_averages().table(sort_by="cpu_time_total")
                            return
                    self.training_logs.append({'epoch': epoch, 'step': step, 'loss': loss.item()})
                    progress_bar.update(1)
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
        
        self.train_time = time.time() - train_start_time
    
    def evaluate(self):
        self.model.eval()
        # metric = evaluate.load("accuracy")
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                ground_truth = batch['labels']
                correct += (predictions == ground_truth).sum().item()
                total += len(ground_truth)
        self.model.train()
        return correct / total
        
