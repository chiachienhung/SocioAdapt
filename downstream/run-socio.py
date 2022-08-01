import os
from dataclasses import dataclass, field
from typing import Optional
import logging
from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelWithHeads
from transformers import TrainingArguments, Trainer, EvalPrediction, HfArgumentParser, MODEL_FOR_MASKED_LM_MAPPING, MultiLingAdapterArguments, AdapterTrainer
from transformers.trainer_callback import EarlyStoppingCallback
import math 
import torch

os.environ["WANDB_DISABLED"] = "true"
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pretaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The adapter checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    ) 
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
        default=False,
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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    multisource_data: Optional[str] = field(
        default=None, metadata={"help": "JSON multi-source dataset descriptor."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
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
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()


# Load dataset
dataset = load_dataset('./trustpilot.py', name=data_args.dataset_name, cache_dir="./reload")

label_list = dataset["train"].unique("label")
label_list.sort()
num_labels = len(label_list)

if model_args.tokenizer_name:
    tokenizer_model = model_args.tokenizer_name
else:
    tokenizer_model = model_args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, cache_dir=model_args.cache_dir)
padding = "max_length" if data_args.pad_to_max_length else False

if data_args.max_seq_length is None:
    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 1024:
        logger.warn(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
        )
        max_seq_length = 1024
else:
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["review"], max_length=max_seq_length, truncation=True, padding=padding)

dataset = dataset.map(encode_batch, batched=True, remove_columns=['review'])
dataset = dataset.shuffle(seed=42)

train_dataset = dataset['train']
eval_dataset = dataset["validation"]
test_dataset = dataset['test']


if data_args.max_train_samples:
    train_dataset = train_dataset.select(range(data_args.max_train_samples))
if data_args.max_eval_samples:
    eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
if data_args.max_test_samples:
    test_dataset = test_dataset.select(range(data_args.max_test_samples))
      
# Load config & model

config = AutoConfig.from_pretrained(
    model_args.model_name_or_path, num_labels=num_labels, cache_dir=model_args.cache_dir
)
model = AutoModelWithHeads.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir
)

if adapter_args.train_adapter:
    print("Training adapter")
    #adapter_name = model.load_adapter(adapter_name_or_path, config="pfeiffer", load_as = 'MLM')
    model.load_adapter(model_args.adapter_name_or_path, config="pfeiffer", load_as = 'MLM')
    model.add_classification_head(
        "MLM",
        num_labels=num_labels,
        overwrite_ok=True
      )
    model.train_adapter(['MLM'])
    model.set_active_adapters(['MLM'])
    task = data_args.dataset_name  + ".adapter"

else:
    print("Not training adapter")
    model.add_classification_head(
        "socio",
        num_labels=num_labels,
      )
    task = data_args.dataset_name

callbacks = [EarlyStoppingCallback(model_args.patience, 0.00001)]
training_args.remove_unused_columns=False
print(model_args, data_args, training_args, adapter_args)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    acc_metric = load_metric("accuracy")
    acc = acc_metric.compute(predictions=preds, references=p.label_ids)
    f1_metric = load_metric("f1")
    f1_macro = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")['f1']
    f1_weighted = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")['f1']
    return {"accuracy": acc['accuracy'], "f1_macro": f1_macro, "f1_weighted": f1_weighted}

if adapter_args.train_adapter:
    trainer_class = AdapterTrainer 
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )
else:
    trainer_class = Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )

###TRAIN###
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

###EVAL###
if training_args.do_eval:
    if adapter_args.train_adapter:
        trainer.model.heads.to("cuda")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    perplexity = math.exp(metrics["eval_loss"])
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# ###TEST###
if training_args.do_predict:
    #predictions = trainer.predict(test_dataset=test_dataset).predictions
    #print(predictions)
    metrics = trainer.predict(test_dataset=test_dataset).metrics
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    metrics["predict_samples"] = len(test_dataset)
    trainer.log_metrics("predict_1", metrics)
    trainer.save_metrics("predict_1", metrics)
    output_test_file = os.path.join(training_args.output_dir, f"test_results_1_{task}.txt")

    with open(output_test_file, "w") as writer:
        writer.write("index\tprediction\tlabel\n")
        for index, item in enumerate(predictions):
            item = label_list[item]
            true_label = test_dataset[index]['label']
            writer.write(f"{index}\t{item}\t{true_label}\n")

if training_args.do_predict:
    model = AutoModelWithHeads.from_pretrained(
        training_args.output_dir,
        config=config,
        cache_dir=model_args.cache_dir
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )
    #predictions = trainer.predict(test_dataset=test_dataset).predictions
    #print(predictions)
    metrics = trainer.predict(test_dataset=test_dataset).metrics
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    metrics["predict_samples"] = len(test_dataset)
    trainer.log_metrics("predict_2", metrics)
    trainer.save_metrics("predict_2", metrics)
    output_test_file = os.path.join(training_args.output_dir, f"test_results_2_{task}.txt")

    with open(output_test_file, "w") as writer:
        writer.write("index\tprediction\tlabel\n")
        for index, item in enumerate(predictions):
            item = label_list[item]
            true_label = test_dataset[index]['label']
            writer.write(f"{index}\t{item}\t{true_label}\n")