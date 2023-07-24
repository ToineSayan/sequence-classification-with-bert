from typing import List, Optional
from dataclasses import dataclass, field



SEQUENCE_CLASSIFICATION = "seq_classification"
TOKEN_CLASSIFICATION = "token_classification"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(
        default='', metadata={"help": "The dataset used for pre-training."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    task_inputs_col: str = field(
        default='', 
        metadata={"help": "The column name in which inputs are stored in the data CSV file."}
    )
    task_labels_col: str = field(
        default='', 
        metadata={"help": "The column name in which labels to be predicted are stored in the data CSV file."}
    )

    path_base: Optional[str] = field(
        default=None, metadata={"help": "base path to the dataset"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the test data."}
    )
    tokenizer_path: Optional[str] = field(
        default=None, metadata={"help": "The base model used to initialize the tokenizer"}
    )

    fill_label_value_strategy: Optional[int] = field(
        default=None, metadata={"help": "The strategy to apply when a label is undefined." 
                                "If None will assign a random label else fills the labels with the value provided"}
    )
    multilabel_classification: str = field(
        default=False, metadata={"help": "True if the final task a multilabel problem"}
    )
    other_args: dict = field(
        default_factory=dict, metadata={"help": "Other useful parameters for some specific datasets"}
    )

    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Activates the debug mode if True (reduces the size of the train, validation, and test sets for debugging purpose)"}
    )
    debug_split_sizes: Optional[int] = field(
        default=100, metadata={"help": "Used only if debug_mode is True, size of the subsets of train, validation, and test sets"}
    )
    




