import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

class ProfilingTrainer:
    def __init__(
            self, model, train_dataloader, val_dataloader, optimizers, 
            device, n_steps_per_val, target_val_acc
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
        
    def train(self, n_epochs):
        self.model.to(self.device)
        self.optimizer.zero_grad()
        train_start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("forward"):
                        outputs = self.model(**batch)
                    with record_function("backward"):
                        loss = outputs[0]
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                if (step + 1) % self.n_steps_per_val == 0:
                    val_acc = self.evaluate()
                    self.val_logs.append({'step': step, 'accuracy': val_acc})
                    if val_acc >= self.target_val_acc:
                        print(f"Stopping training at epoch {epoch+1}, step {step+1} as target validation accuracy reached")
                        self.train_time = time.time() - train_start_time
                        self.profiler_log = prof.key_averages().table(sort_by="cpu_time_total")
                        return
                
                self.training_logs.append({'epoch': epoch, 'step': step, 'loss': loss.item()})
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
        
        self.train_time = time.time() - train_start_time
        self.profiler_log = prof.key_averages().table(sort_by="cpu_time_total")
    
    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                total_correct += torch.sum(preds == labels)
                total += len(labels)
        return total_correct / total
