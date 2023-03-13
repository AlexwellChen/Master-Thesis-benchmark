#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The LightSeq Team
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import copy
import torch
import numpy as np
# from datasets import load_dataset, load_metric
import datasets
from trainer_accelerate import AcceleratorTrainer
from adan import Adan
from accelerate import Accelerator

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForSequenceClassification,
    BertLayer,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from ls_hf_transformer_layer import inject_ls_layer, LSBertForSequenceClassification



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {
    "imdb": ("text", None),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    module_type: int = field(
        default=1,
        metadata={
            "help": "0: original Hugging Face layer, 1: LightSeq CUDA layer, 2: custom Torch layer"
        },
    )
    enable_quant: bool = field(
        default=False,
        metadata={"help": "Whether to enable quantization"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=2,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    # Replace with lightseq encoder layers and save the huggingface model
    model = LSBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        training_args=training_args,
        model_args=model_args,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16)
    
    lr = 1e-4
    betas = (0.98, 0.92, 0.99)
    wd = 0.01

    accelerator = Accelerator()
    optimizer = Adan(params=model.parameters(), lr=lr, weight_decay=wd, fused=True, foreach=True, betas=betas, eps=1e-8)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=50, 
                                                num_training_steps=len(train_loader) * 2
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
        n_steps_per_val=50,
        target_val_acc=0.93,
        log_file_name="imdb_adan_fused_lr1e-4_wd1e-2_wm50_ep2_acc93_ls",
        seed=38,
    )

    trainer.train(2)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
