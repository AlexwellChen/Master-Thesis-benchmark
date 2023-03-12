import time
import torch
from tqdm import tqdm
import pynvml


# torch.manual_seed(38)

class AcceleratorTrainer:
    def __init__(
            self, model, accelerator, train_dataloader, val_dataloader, test_dataloader, optimizers, 
            device, n_steps_per_val, target_val_acc, log_file_name='default_log'
        ):
        self.model = model
        self.accelerator = accelerator
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizers[0]
        self.scheduler = optimizers[1]
        self.device = device
        self.n_steps_per_val = n_steps_per_val
        self.target_val_acc = target_val_acc
        self.training_logs = []
        self.val_logs = []
        self.train_time = 0
        self.log_file_name = log_file_name
        self.device_count = 0
        self.total_energy = 0
        self.sm_occupancy = []
        self.avg_sm_occupancy = 0

        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def get_sm_occupancy(self):
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Get SM occupancy
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.sm_occupancy.append(info.gpu)
    
    def get_energy(self):
        # Use this func for AMDs GPU, TPUs, etc
        pass


    #@measure_energy()
    def train(self, n_epochs):
        self.model.to(self.device)
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(range(n_epochs * len(self.train_dataloader)), desc="Epoch")
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+self.log_file_name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        acc_achieved = 0
        train_start_time = time.time()
        with prof:
            for epoch in range(n_epochs):
                epoch_start_time = time.time()
                self.model.train()
                for step, batch in enumerate(self.train_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.model.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    prof.step()
                    self.get_sm_occupancy()
                    if (step + 1) % self.n_steps_per_val == 0:
                        val_acc = self.evaluate()
                        self.val_logs.append({'step': step, 'accuracy': val_acc})
                        if self.target_val_acc is not None:
                            print(f"Validation accuracy at step {step+1}: {val_acc:.4f}, loss: {loss.item():.4f}, target: {self.target_val_acc:.2f}")
                        else:
                            print(f"Validation accuracy at step {step+1}: {val_acc:.4f}, loss: {loss.item():.4f}")
                        if self.target_val_acc is not None and val_acc >= self.target_val_acc:
                            if acc_achieved == 2: 
                                print(f"Stopping training at epoch {epoch+1}, step {step+1} as target validation accuracy reached")
                                self.train_time = time.time() - train_start_time
                                # average sm occupancy
                                self.avg_sm_occupancy = sum(self.sm_occupancy) / len(self.sm_occupancy)
                                return
                            else:
                                acc_achieved += 1
                                print("Target validation accuracy reached, " + str(3-acc_achieved), " more times to stop training")
                    self.training_logs.append({'epoch': epoch, 'step': step, 'loss': loss.item()})
                    progress_bar.update(1)
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
        self.train_time = time.time() - train_start_time
        # average sm occupancy
        self.avg_sm_occupancy = sum(self.sm_occupancy) / len(self.sm_occupancy)
        
        
    def evaluate(self):
        self.model.eval()
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

    def test(self):
        if self.test_dataloader is None:
            raise ValueError("No test dataloader provided.")
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                ground_truth = batch['labels']
                correct += (predictions == ground_truth).sum().item()
                total += len(ground_truth)
        self.model.train()
        return correct / total
        
