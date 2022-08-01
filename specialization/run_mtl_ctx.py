import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from mtl_model_ctx import MultiTaskModel, MultiTaskModelWeight
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    set_seed, 
    DataCollatorForTokenClassification, 
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import get_last_checkpoint
import torch
import numpy as np

import random
os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: "},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    patience: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of epochs for early stopping criteria."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the dataset cache"},
    )    
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
                
@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int

def tokenize_MLM_dataset(
    raw_datasets,
    tokenizer,
    task_id,
    text_column_name,
    data_args,
    training_args,
):
    
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
    raw_datasets = raw_datasets.map(lambda example: {'labels': [-1]*(max_seq_length)})
    # When using line_by_line, we just tokenize each nonempty line.
    if data_args.line_by_line:
        # Padding strategy
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            result = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
            
            result["task_ids"] = [task_id]*len(examples[text_column_name])
            return result

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=[text_column_name],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            result["task_ids"] = [task_id]*len(concatenated_examples)
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
    return tokenized_datasets


def load_MLM_dataset(task_id, tokenizer, data_args, training_args, model_args):

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    raw_datasets = raw_datasets.map(lambda example: {'text': example['text'].split('\t')[1]})
    
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = tokenize_MLM_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        text_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name="mlm",
        num_labels=0,
        type="mask_language_modeling",
    )
    tokenized_datasets.shuffle(3)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )

def tokenize_seq_classification_dataset(
    tokenizer, raw_datasets, task_id, text_column_name, data_args, training_args
):
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False # Pad later, dynamically at batch creation, to the max sequence length in each batch
    
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    raw_datasets = raw_datasets.map(lambda example: {'label': [example['label']]+[0]*(max_seq_length-1)})
    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        result = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
        examples["labels"] = examples.pop("label")
        result["task_ids"] = [task_id] * len(examples["labels"])

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=[text_column_name],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets


def load_seq_classification_dataset(task_id, tokenizer, data_args, training_args, model_args):

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    all_labels = ['F', 'M'] ### need to remember the labels here ['F', 'M'], ['0', '2']
    label_2_id = {k: v for v, k in enumerate(all_labels)}
    
    ## Whether to convert label to id
    raw_datasets = raw_datasets.map(lambda example: {'label': label_2_id[example['text'].split('\t')[0]]})
    raw_datasets = raw_datasets.map(lambda example: {'text': example['text'].split('\t')[1]})

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = tokenize_seq_classification_dataset(
        tokenizer,
        raw_datasets,
        task_id,
        text_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id, name="seqclass", num_labels=2, type="mask_seq_classification"
    )
    tokenized_datasets.shuffle(10)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )

def load_mtl_datasets(tokenizer, data_args, training_args, model_args):
    (
        seq_classification_train_dataset,
        seq_classification_validation_dataset,
        seq_classification_task,
    ) = load_seq_classification_dataset(1, tokenizer, data_args, training_args, model_args)
    (
        mlm_train_dataset,
        mlm_validation_dataset,
        mlm_task,
    ) = load_MLM_dataset(0, tokenizer, data_args, training_args, model_args)
    
    train_dataset = datasets.concatenate_datasets([mlm_train_dataset, seq_classification_train_dataset])
    train_dataset.shuffle(seed=123)
    #validation_dataset = datasets.concatenate_datasets([seq_classification_validation_dataset, mlm_validation_dataset])
    validation_dataset = [
        seq_classification_validation_dataset,
        mlm_validation_dataset,
    ]
    check_dataset = datasets.concatenate_datasets([mlm_validation_dataset, seq_classification_validation_dataset])

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "eval": check_dataset}
    )
    tasks = [seq_classification_task, mlm_task]
    return tasks, dataset

InputDataClass = NewType("InputDataClass", Any)

seq_data_collator = default_data_collator
@dataclass
class DataCollatorForMTL:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, examples: List[dict]):
        batch = self.collate_batch(examples)
        return batch

    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        mlm_result = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).torch_call(features)

        textclass_result = seq_data_collator(features)
        textclass_result.pop('special_tokens_mask', None)

        index_0 = torch.nonzero(textclass_result['task_ids'] == 0).view(-1).detach().numpy()
        if len(index_0)>0:
            for s in index_0:
                for k, v in textclass_result.items():
                    textclass_result[k][s]=mlm_result[k][s]
        index_1 = torch.nonzero(textclass_result['task_ids'] == 1).view(-1).detach().numpy()
        if len(index_1)>0:
            for s in index_1:
                for k, v in textclass_result.items():
                    if k!='labels':
                        textclass_result[k][s]=mlm_result[k][s]
        
        return textclass_result

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    if preds.ndim == 2:
        # Sequence classification
        preds = np.argmax(preds, axis=1)
        if p.label_ids.ndim != 1:
            gold_labels = p.label_ids[:, 0]
        else:
            gold_labels = p.label_ids
        return {"accuracy": (preds == gold_labels).astype(np.float32).mean().item()}
    else:
        raise NotImplementedError()

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    print(model_args, training_args, data_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
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

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    mask_ids = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
    tasks, raw_datasets = load_mtl_datasets(tokenizer, data_args, training_args, model_args)

    model = MultiTaskModelWeight(model_args.model_name_or_path, model_args.cache_dir, mask_ids, tasks)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(data_args.max_eval_samples)))

            eval_datasets = new_ds
    print(eval_datasets)
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForMTL(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None)

    check_dataset = raw_datasets["eval"]
    callbacks = [EarlyStoppingCallback(model_args.patience, 0.0001)]
    # Initialize our Trainer
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset if training_args.do_train else None,
       eval_dataset=check_dataset if training_args.do_eval else None,
       compute_metrics=None,
       tokenizer=tokenizer,
       data_collator=data_collator,
       callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        checkpoint=None ##Can later modify it
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
           data_args.max_train_samples
           if data_args.max_train_samples is not None
           else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        model.encoder.save_pretrained(training_args.output_dir + '/only_encoder')

    # Evaluation
    if training_args.do_eval:
        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "mask_language_modeling":
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm_probability=data_args.mlm_probability,
                    pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            )
                trainer.compute_metrics=None
                trainer.data_collator = data_collator
                logger.info("*** Evaluate ***")

                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                try:
                    perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                    perplexity = float("inf")
                metrics["perplexity"] = perplexity

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
            else:
                if data_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                       tokenizer, pad_to_multiple_of=8
                   )
                else:
                    data_collator = None
                trainer.compute_metrics = compute_metrics
                trainer.data_collator = data_collator
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                   data_args.max_eval_samples
                   if data_args.max_eval_samples is not None
                   else len(eval_datasets)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
            
if __name__ == "__main__":
    main()
