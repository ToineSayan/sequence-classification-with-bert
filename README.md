
# Document Classification via Bert

## Models

### Bert for Sequence classification (*class transformers.BertForSequenceClassification*)

Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output)

Useful links: 
- Class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification) from the Transformers Library 

  
### Bert for Sequence classification with word attention

Bert Model transformer with a sequence classification head on top (a layer with word attention on the tokens of the sequence (CLS included))

Implementation of section "2.2 Hierarchical Attention > Word Attention" in [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174.pdf)
Adaptation of the class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification)

Useful links: 
- Article: [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174.pdf)
- Class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification) from the Transformers Library 

## Datasets

  
### Presentation

**WikiVitals (level 4)**
* *Description (from NetSet):* Vital articles of Wikipedia in English (level 4) with [...] words used in summaries (tokenization by Spacy, model "en_core_web_lg").
* Task: classification of the articles according to their topic. Each article has 1 or more label that corresponds to a unique path in a hierarchy of labels
	* possible tasks: single-label classification, multilabel classification
* *Dataset infos and download:* [NetSet - WikiVitals (en)](https://netset.telecom-paris.fr/pages/wikivitals.html) (texts not available)
* *Source:* [Wikivitals Level 4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4) (the source has changed since the dataset creation in June 2021)

**Web of Science**
* *Description:* 
* *Task:* classification of the articles according to their topic. Each article has 2 labels that corresponds to a unique path in a hierarchy of labels
	* possible tasks: single-label classification, multilabel classification
* *Dataset infos and download:* 
* *Source:* 

### Statistics

Multilabel datasets:

| Dataset name                 | # total | # train | # val. | # test | max. (& avg.) depth | # labels | Type of classification          |
| ---------------------------- | ------- | ------- | ------ | ------ | ------------------- | -------- | ------------------------------- |
| WoS (Web of Science)         |         | 30,070  | 7,518  | 9,397  | 2 (2.0)             | 141      | Article classification by topic |
| wikivitals-lvl4              | 10,011  | 6,407   | 1,602  | 2,003  | 3 (--)              | 587      | Article classification by topic |
| wikivitals-lvl5              |         |         |        |        |                     |          | Article classification by topic |
| wikivitals-lvl5 (my version) |         |         |        |        |                     |          | Article classification by topic |

## Training

To do a training (or an evaluation), run the following command:

```shell
# Training of a model using the configuration file named config_{dataset_id}.yml
> python classification.py --c dataset_id
# Evaluation of a model using the configuration file named config_{dataset_id}.yml
> python classification.py --c dataset_id --evaluate_only True
```

**Training steps:**
Training can be performed in 1 or 2 steps. 
Steps are the following ones and, if both performed, will occur in this order:
1) Training of the classification head only (Bert model used is frozen)
2) Training of the Bert model and the classification head 
Each step is optional. 
To deactivate a training step, set the parameter 'do_train' to False in the appropriate training configurations (in the configuration file)

**Note on evaluation:** 
To evaluate a model on a train set and a validation set, one can use the --evaluate_only argument or deactivate the 2 training steps in a configuration file

**Parameters for training:**
See this

## Results (on the test set)

### Single label classification

to-do

### Multilabel classification

w/ transformers.BertForSequenceClassification

| Dataset name         | max. # tokens    | micro-F1       | macro-F1       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | -------------- | ---------------- | ------------------------------ |
| WoS (Web of Science) | 512              | 84.96          | 76.72          | wos              | Consistent w/ results in X & X |
| WikiVitals (level 4) | 16               | 72.69          | 19.48          | wikivitals       |                                |
| WikiVitals (level 4) | 128              | 85.74          | 37.36          | wikivitals       |                                |
| WikiVitals (level 4) | 512              | 85.99          | 34.40          | wikivitals       |                                |


w/ BertWithWordAttention

| Dataset name         | max. # tokens    | micro-F1       | macro-F1       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | -------------- | ---------------- | ------------------------------ |
| WoS (Web of Science) | 512              |                |                | wos              |                                |
| WikiVitals (level 4) | 16               |                |                | wikivitals       |                                |
| WikiVitals (level 4) | 128              | 85.94          | 36.18          | wikivitals       |                                |
| WikiVitals (level 4) | 512              |                |                | wikivitals       |                                |
