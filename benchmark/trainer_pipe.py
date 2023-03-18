import time
import torch
from tqdm import tqdm
import pynvml
# F.cross_entropy
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

class PipelineTrainer:
    def __init__(
            self, model, train_dataloader, val_dataloader, test_dataloader, optimizers, 
            device, n_steps_per_val, target_val_acc, log_file_name='default_log', seed=42
        ):
        self.model = model
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
        torch.manual_seed(seed)

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
        scaler = GradScaler()
        self.optimizer.zero_grad()
        in_device = self.model.devices[0]
        out_device = self.model.devices[-1]
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
                step = 0
                for input, target in self.train_dataloader:
                    input = input.to(in_device)
                    target = target.to(out_device)
                    self.model.zero_grad()
                    
                    with autocast():
                        outputs = self.model(input)
                        loss = F.cross_entropy(outputs, target)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    prof.step()
                    self.get_sm_occupancy()
                    if (step + 1) % self.n_steps_per_val == 0:
                        val_acc = self.evaluate()
                        self.val_logs.append({'step': step, 'accuracy': val_acc})
                        print(f"Validation accuracy at step {step+1}: {val_acc:.4f}, loss: {loss.item():.4f}, target: {self.target_val_acc:.2f}")
                        if self.target_val_acc is not None and val_acc >= self.target_val_acc:
                            if acc_achieved == 2: 
                                print(f"Stopping training at epoch {epoch+1}, step {step+1} as target validation accuracy reached")
                                self.train_time = time.time() - train_start_time
                                # average sm occupancy
                                self.avg_sm_occupancy = sum(self.sm_occupancy) / len(self.sm_occupancy)
                                return
                            else:
                                acc_achieved += 1
                                step += 1
                                print("Target validation accuracy reached, " + str(3-acc_achieved), " more times to stop training")
                    self.training_logs.append({'epoch': epoch, 'step': step, 'loss': loss.item()})
                    progress_bar.update(1)
                    step += 1
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
        in_device = self.model.devices[0]
        out_device = self.model.devices[-1]
        with torch.no_grad():
            for input, target in self.train_dataloader:
                input = input.to(in_device)
                target = target.to(out_device)
                self.model.zero_grad()
                outputs = self.model(input)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == target).sum().item()
                total += len(target)
        self.model.train()
        return correct / total

    def test(self):
        if self.test_dataloader is None:
            raise ValueError("No test dataloader provided.")
        self.model.eval()
        correct = 0
        total = 0
        in_device = self.model.devices[0]
        out_device = self.model.devices[-1]
        with torch.no_grad():
            for input, target in self.train_dataloader:
                input = input.to(in_device)
                target = target.to(out_device)
                self.model.zero_grad()
                outputs = self.model(input)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == target).sum().item()
                total += len(target)
        self.model.train()
        return correct / total
        
