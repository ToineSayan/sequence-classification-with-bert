import yaml

from transformers import TrainingArguments, BertConfig
from transformers import  BertTokenizer, Trainer, DefaultDataCollator
from transformers import BertForSequenceClassification
from sequence_classification import BertForSequenceClassificationWithWordAttention

from arguments import DataTrainingArguments

import torch
import random
import numpy as np
import argparse

import copy


# Parsing configurations
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, required=True)
parser.add_argument('--eval_only', type=bool, required=False)
parsed_args = parser.parse_args()
config_file = './configurations/config_' + parsed_args.c + '.yml'
print(config_file)
# logging.set_verbosity_info()

# Load params for data preprocessing, model and trainer
with open(config_file, 'r') as stream:
    args = yaml.safe_load(stream)

# set seed for reproductibility if provided
try: 
    global_seed = args["training_args"]["seed"]
except:
    global_seed = random.randint(0, 1000) 
torch.manual_seed(global_seed)
random.seed(global_seed)
np.random.seed(global_seed)


# Model config : BertConfig -> https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/bert#transformers.BertConfig
#       inherits PretrainedConfig -> https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/configuration#transformers.PretrainedConfig 
# Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. 
# Use from_pretrained() to load the model weights.


# Trainer config : TrainingArguments -> https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
# Data processing config : 

# BertForSequenceClassification : https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/bert#transformers.BertForSequenceClassification
#       Code: https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/bert/modeling_bert.py#L1517


data_args = DataTrainingArguments(**args['data_args'])
model_args = BertConfig(**args['model_args'])
training_args = TrainingArguments(**args['training_args'])
preliminary_training_args = args['preliminary_training_args']
# Get the configuration for the 'head only training' phase 
head_only_training_args = copy.deepcopy(training_args)
if not preliminary_training_args == None:
    for k,v in preliminary_training_args.items():
        head_only_training_args.__setattr__(k,v)
else:
    head_only_training_args.do_train = False
if parsed_args.eval_only:
    training_args.do_train = False
    head_only_training_args.do_train = False



# Use the appropriate tokenizer for data pre-processing
data_args.tokenizer_path = model_args.name_or_path 
# Set the problem type
model_args.problem_type = "multi_label_classification" if data_args.multilabel_classification else "single_label_classification"

# -----------------------------------------------------------------
# Load and prepare dataset
# -----------------------------------------------------------------
from load_datasets import load_dataset_by_name
tokenizer = BertTokenizer.from_pretrained(data_args.tokenizer_path)
# Display data pre-processing configuration
print(f"Data pre-processing configuration:\n{data_args}\n")
# Pre-process the data (includes tokenization + padding + truncation)
preprocessed_dataset, label2id = load_dataset_by_name(
    data_args=data_args, 
    tokenizer=tokenizer
    )
# Collate data
data_collator = DefaultDataCollator(
    return_tensors='pt'
)

# -----------------------------------------------------------------
# Set up the model
# -----------------------------------------------------------------
# Update the model configuration
model_args.num_labels = len(label2id.keys())
model_args.label2id = label2id
model_args.id2label = {str(v): k for k,v in label2id.items()}
# Display model configuration
print(f"\nModel configuration:\n{model_args}\n")
# Initialize the model
if model_args.classification_head == 'base_sequence_classifier':
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_args.name_or_path, config=model_args)
elif model_args.classification_head == 'sequence_classifier_with_word_attention':
    model = BertForSequenceClassificationWithWordAttention.from_pretrained(pretrained_model_name_or_path=model_args.name_or_path, config=model_args)
else:
    raise ValueError("Unknown classification head, must be in ['base_sequence_classifier', 'sequence_classifier_with_word_attention']")
# Define a function to count the number of trainable parameter in the model
def count_parameters(model_to_evaluate):
    return sum(p.numel() for p in model_to_evaluate.parameters() if p.requires_grad)

# -----------------------------------------------------------------
# Training and saving the model
# -----------------------------------------------------------------
# Preliminary training
if head_only_training_args.do_train:
    # Do a preliminary training phase where only the classification head is trained
    print("\nPreliminary training starts...")
    # Display training configuration
    print(f"\nPreliminbary training configuration:{training_args}\n")
    # freeze Bert in the model
    for param in model.bert.parameters():
        param.requires_grad = False
    # Display the number of trainable parameters
    print(f"Number of trainable parameters (head only training phase): {count_parameters(model)}\n")
    # Initialize the trainer for the preliminary training
    trainer = Trainer(
        model=model,
        args=head_only_training_args, # TrainingArguments object https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/trainer#transformers.TrainingArguments
        train_dataset=preprocessed_dataset['train'],
        eval_dataset=preprocessed_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    # Train the model
    trainer.train()


# Unfreeze Bert (has no effect if Bert has not been frozen)
for param in model.bert.parameters():
    param.requires_grad = True
# Initialize the trainer for full training
trainer = Trainer(
    model=model,
    args=training_args, # TrainingArguments object https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/trainer#transformers.TrainingArguments
    train_dataset=preprocessed_dataset['train'],
    eval_dataset=preprocessed_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)
if training_args.do_train:
    # Full training
    print("\nFull training starts...")
    # Display training configuration
    print(f"\nTraining configuration:{training_args}\n")
    print(f"Number of trainable parameters (full training): {count_parameters(model)}\n")
    # Train the model
    trainer.train() if training_args.do_train else None


# Save the model
import time
trainer.save_model(f"./outputs/final_models/model_{parsed_args.c}_{time.time()}")


# -----------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------
print("\nEvaluation starts...")
import evaluate
# Define the metrics to compute for each problem type (single label or multilabel classification)
def compute_metrics(eval_preds, threshold = 0.5, problem_type = model_args.problem_type): 
    logits, labels = eval_preds
    if problem_type == "single_label_classification" :
        # single label classification
        ptype = None
        predictions = np.argmax(logits, axis=-1).reshape(-1,1)
        labels_ = labels
        metrics = ["accuracy", "micro-f1", "macro-f1"]
    elif problem_type ==  "multi_label_classification":
        # multi label classification
        ptype = "multilabel"
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs > threshold)] = 1
        predictions = predictions.astype('int32')
        labels_ = labels.astype('int32')
        # labels_ = labels
        metrics = ["micro-f1", "macro-f1"]
    else: 
        raise ValueError("Wrong problem type")
    # Compute the output
    outputs = dict()
    if "accuracy" in metrics:
        metric = evaluate.load("accuracy")
        accuracy = metric.compute(predictions=predictions, references=labels_)
        outputs["accuracy"] = accuracy["accuracy"]
    if "micro-f1" in metrics:
        metric = evaluate.load("f1", ptype)
        f1_micro = metric.compute(predictions=predictions, references=labels_, average = 'micro')
        outputs["micro-f1"] = f1_micro["f1"]
    if "macro-f1" in metrics:
        metric = evaluate.load("f1",  ptype)
        f1_macro = metric.compute(predictions=predictions, references=labels_, average = 'macro')
        outputs["macro-f1"] = f1_macro["f1"]
    return outputs
# Update the compute metrics module of the trainer
trainer.compute_metrics = compute_metrics
# Do inference on the train and test sets 
with torch.no_grad(): # from here, only inference 
    # results_train = trainer.predict(preprocessed_dataset['train'], metric_key_prefix='train')
    # print(results_train[-1])
    results_test = trainer.predict(preprocessed_dataset['test'], metric_key_prefix='test')
    print(results_test[-1])


