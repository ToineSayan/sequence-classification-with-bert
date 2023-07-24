from pathlib import Path

from torch import Value
from datasets import load_dataset, Features, Value, Sequence
import random
# from datasets_utils import get_label2id, get_id2label
import warnings


def load_dataset_by_name(data_args, tokenizer):
    if data_args.dataset == 'wikivitals-telecom': # wikivitals level 4
        return(load_wv_or_wos(data_args, tokenizer))
    elif data_args.dataset == 'wos': # Web of Science
        return(load_wv_or_wos(data_args, tokenizer)) # normal - même structure


def load_wv_or_wos(
        data_args,
        tokenizer):
    

    data_files = {
        'train': f"{data_args.path_base}{data_args.train_file}",
        'validation': f"{data_args.path_base}{data_args.validation_file}",
        'test': f"{data_args.path_base}{data_args.test_file}",
    }

    ds = load_dataset('json', data_files=data_files)

    # ---------------------------------------------------------------------
    # Infos
    print(f"Dataset name: {data_args.dataset}")
    print(f"Original number of training samples: {len(ds['train'])}")
    print(f"Original number of validation samples: {len(ds['validation'])}")
    print(f"Original number of test samples: {len(ds['test'])}")
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # DEBUG MODE - SELECT A SUBSET OF THE DATASET
    if data_args.debug_mode:
        print("DEBUG MODE - Activated")
        print(f"DEBUG MODE - Reducing the size of the sets to {data_args.debug_split_sizes} samples in each of them")
        for split in ['train', 'validation', 'test']:
            ds[split] = ds[split].select(list(range(data_args.debug_split_sizes)))
    # ---------------------------------------------------------------------
   
    
    inputs_col, labels_col = data_args.task_inputs_col, data_args.task_labels_col
    ds = ds.map(lambda example: {'texts': example[inputs_col][0]}, load_from_cache_file=False)

    # In this dataset, articles are classified according to a hierarchical classification.
    # In the original file, labels are stored as follow : ['A', 'child_of_A', 'grandchild_of_A', ...]. 
    # The following line renames the labels in : ['A', 'A ->- child_of_A', 'A ->- child_of_A ->- grandchild_of_A', ...] so that each labels contains information about the hierarchy from the root.
    ds = ds.map(lambda example: {'labels': [' ->- '.join(example[labels_col][:i+1]) for i in range(len(example[labels_col]))]}, load_from_cache_file=False)

    ds = ds.remove_columns([inputs_col, 'topic', 'keyword', labels_col])


    # Filter the labels by depth if set
    if 'depth' in data_args.other_args.keys() and data_args.other_args['depth'] is not None:
        depth = data_args.other_args['depth']
        ds = ds.map(
            lambda example: {
                    'labels' : example['labels'][:depth] if len(example['labels'])>=depth else example['labels']
                }, 
                load_from_cache_file=False
                ) 

    if not data_args.multilabel_classification:
        # If no multilabel, we use the lowest classification level
        ds = ds.map(lambda example: {'labels': [example['labels'][-1]]}, load_from_cache_file=False)

    

   

    labels_list = ds['train']['labels'] + ds['validation']['labels'] + ds['test']['labels']
    labels_list = [inner for outer in labels_list for inner in outer]
    labels_list = sorted(list(set(labels_list)))


    num_labels = len(labels_list)
    print(f"Total number of labels: {num_labels}")
    # Attribute an id to each label
    label2id = {labels_list[i]: i for i in range(num_labels)}



    # Replace labels by their id
    ds = ds.map(lambda example: {'labels': [label2id[example['labels'][i]] for i in range(len(example['labels']))]}, load_from_cache_file=False)
    
    if data_args.multilabel_classification:
        ds = ds.map(lambda example: {'labels': [1 if i in example['labels'] else 0 for i in range(num_labels)]}, load_from_cache_file=False)
        ds = ds.cast(Features({"texts": Value("string"), "labels": Sequence(feature=Value(dtype="float32", id = None))}))


    # print(ds['train'][:20])
    # print(ds['train'].features)

    # with open('labels.txt', 'w') as f:
    #     for l in labels_list:
    #         f.write((l + '\n')

    # quit()






    # Define the padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # # # Initialize the tokenizer. We use the tokenizer corresponding to the base language model used.
    # # tokenizer = BertTokenizer.from_pretrained(data_args.base_model_for_tokenizer, do_lower_case=True)

    # Make sure the max seq length is smaller or equal to model max length (if not, set it to max length)
    if data_args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    

    # # Use the mapping method to tokenize each sentence in our dataset 
    # # This operation sets values for each parameters of elements in our dataset:
    # # * input_ids: token ids to be used as input of the language model
    # # * token_type_ids: token type ids
    # # * attention_mask: attention mask
    def preprocess_function(examples):
        result = tokenizer(examples['texts'], padding=padding, max_length=max_seq_length, truncation=True)
        return result
    
    
    # Mapping https://huggingface.co/docs/datasets/v2.12.0/en/package_reference/main_classes#datasets.Dataset.map
    ds = ds.map(
              preprocess_function,
              batched=True,
              load_from_cache_file= False,
              desc="Running tokenizer on dataset",
          )
    

    return ds, label2id


# NOT TESTED
def load_and_split(data_path, test_size, val_size =  None, shuffle = True, seed = 42):
    ds = load_dataset('json', data_files = data_path, split = "train") # split = train pour récupérer le dataset
    ds = ds.train_test_split(test_size = test_size, shuffle = shuffle, seed = seed)
    if not val_size == None:
        pass
    ds["validation"] = ds["test"]


    
    test =  load_dataset('json', data_files=f"{data_args.path_base}{data_args.test_file}", split="train") # split = train pour récupérer le dataset

    ds["test"] = test

    
