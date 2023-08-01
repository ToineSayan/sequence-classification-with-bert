
# Document or sequence classification via Bert

A constantly evolving document. Calculation of baselines for various datasets used in my NLP research and related projects.
No hyperparameter optimization has been carried out to calculate these results, unless otherwise stated.


## Models

### BSC: Bert for Sequence classification (*class transformers.BertForSequenceClassification*)

Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output)

Useful links: 
- Class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification) from the Transformers Library 

  
### BWA: Bert for Sequence classification with word attention

Bert Model transformer with a sequence classification head on top (a layer with word attention on the tokens of the sequence (CLS included))

Implementation of section "2.2 Hierarchical Attention > Word Attention" in [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174.pdf)
Adaptation of the class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification)

Useful links: 
- Article: [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174.pdf)
- Class [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForSequenceClassification) from the Transformers Library 

## Datasets

### Statistics

Multilabel datasets:

| Dataset name                 | # total | # train | # val. | # test | max. (& avg.) depth | # labels | Type of classification          |
| ---------------------------- | ------- | ------- | ------ | ------ | ------------------- | -------- | ------------------------------- |
| WoS (Web of Science)         | 46,985  | 30,070  | 7,518  | 9,397  | 2 (2.0)             | 141      | Article classification by topic |
| wikivitals-lvl4              | 10,011  | 6,407   | 1,602  | 2,003  | 3 (--)              | 587      | Article classification by topic |
| wikivitals-lvl5              |         |         |        |        |                     |          | Article classification by topic |
| wikivitals-lvl5 (my version) |         |         |        |        |                     |          | Article classification by topic |
  
### WikiVitals (level 4)

*Description (from NetSet):* Vital articles of Wikipedia in English (level 4) with [...] words used in summaries (tokenization by Spacy, model "en_core_web_lg").

* *Associated task:* classification (single-label classification, multilabel classification)
	* classification of the articles according to their topic. Each article has 1 or more label that corresponds to a unique path in a hierarchy of labels
* *Domain:* Research / Education
* *Type:* Real
* *Instance count:* 10,011
* *Data types:* String, Numeric
* *Missing values:* No
* *Dataset infos and download:* [NetSet - WikiVitals (en)](https://netset.telecom-paris.fr/pages/wikivitals.html) (texts not available)
* *Source:* [Wikivitals Level 4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4) (the source has changed since the dataset creation in June 2021)

#### Evaluation

##### Multilabel


| Model                | max. # tokens    | micro-F1       | macro-F1       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | -------------- | ---------------- | ------------------------------ |
| BSC                  | 16               | 72.69          | 19.48          | wikivitals       |                                |
| BSC                  | 128              | 85.74          | 37.36          | wikivitals       |                                |
| BSC                  | 512              | 85.99          | 34.40          | wikivitals       |                                |
| BWA                  | 16               |                |                | wv4_16_BWA       |                                |
| BWA                  | 128              | 85.94          | 36.18          | wv4_128_BWA      |                                |
| BWA                  | 512              | **87.16**      | **37.72**      | wv4_512_BWA      |                                |


### wikiVitals-lvl5-04-2022 (our own)

*Description:* Vital articles of Wikipedia in English (level 5) with words used in summaries.

* *Associated task:* classification (single-label classification, multilabel classification)
	* classification of the articles according to their topic. Each article has 3 labels that corresponds to a unique path in a hierarchy of labels
* *Domain:* Research / Education
* *Type:* Real
* *Instance count:* 48,512
* *Data types:* String, Numeric
* *Missing values:* No
* *Dataset infos and download:* [my Github repo](https://github.com/ToineSayan/wikivitals-lvl5-04-2022) 
* *Source:* complete dump from April, 2022

#### Evaluation

##### Single-label

Split train/validation/test: 81%/9%/10%.
Data split in a stratified way.

Level 0 (11 classes)

| Model                | max. # tokens    | Accuracy       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | ---------------- | ------------------------------ |
| BSC                  | 128              | **95.83**          |  wv-lvl5-04-2022_128_BSC_label0  |                                |
| BSC                  | 512              | 95.17          |   wv-lvl5-04-2022_512_BSC_label0               |                                |
| BWA                  | 128              | 95.57          |  wv-lvl5-04-2022_128_BWA_label0  |                                |

Level 1 (32 classes)


| Model                | max. # tokens    | Accuracy       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | ---------------- | ------------------------------ |
| BSC                  | 128              | **89.42**      | wv-lvl5-04-2022_128_BSC_label1                 |                                |
| BSC                  | 512              |                |                  |                                |
| GMNN w/ FAGCN        | --               | 87.92 (0.31)   |                  | using 0/1 valued representations|

Level 2 (251 classes)

| Method               | max. # tokens    | Accuracy       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | ---------------- | ------------------------------ |
| BSC                  | 128              |                |                  |                                |
| BSC                  | 512              |                |                  |                                |


### Web of Science

*Description:* 

* *Associated task:* classification (single-label classification, multilabel classification)
	* classification of the articles according to their topic. Each article has 2 labels that corresponds to a unique path in a hierarchy of labels
* *Domain:* Research / Education
* *Type:* Real
* *Instance count:* 46,985
* *Data types:* String, Numeric
* *Missing values:* No
* *Dataset infos and download:* to be completed


#### Evaluation

##### Multilabel


| Method               | max. # tokens    | micro-F1       | macro-F1       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | -------------- | ---------------- | ------------------------------ |
|  BSC                 | 512              | 85.51          | 78.10          | wos              |                                |
|  BSC                 | 512              | **86.33**      | 76.77          | wos              |                                |
|  BWA                 | 512              |                |                | wos              |                                |
|  Wang et al. (2022)  | 512              | 85.63          | 79.07          | wos              |                                |
|  Chen et al. (2021)  | 512              | 86.26          | **80.58**      | wos              |                                |

References:
* Haibin Chen, Qianli Ma, Zhenxi Lin, and Jiangyue Yan. 2021. [Hierarchy-aware label semantics matching network for hierarchical text classification.](https://aclanthology.org/2021.acl-long.337/) In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4370–4379, Online. Association for Computational Linguistics.
* Zihan Wang, Peiyi Wang, Lianzhe Huang, Xin Sun, and Houfeng Wang. 2022. [Incorporating hierarchy into text encoder: a contrastive learning approach for hierarchical text classification.](https://aclanthology.org/2022.acl-long.491/) In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 7109–7119. Association for Computational Linguistics.


### EEEC - Enriched Equity Evaluation Corpus

*Description:* EEC (Equity Evaluation Corpus) (Kiritchenko and Mohammad 2018) is a benchmark data set, designed for examining inappropriate biases in system predictions, and it consists of 8,640 English sentences chosen to tease out Racial and Gender related bias. Each sentence is labeled for the mood state it conveys, a task also known as Profile of Mood States (POMS). Each of the sentences in the data set is composed using one of eleven templates, with placeholders for a person’s name and the emotion it conveys. Designed as a bias detection benchmark, the sentences in EEC are very concise, which can make them not useful as training examples. If a classifier sees in training only a small number of examples, which differ only by the name of the person and the emotion word, it could easily memorize a mapping between emotion words and labels, and will not learn anything else. To solve this and create a more representative and natural data set for training, we expand the EEC data set, creating an enriched data set which we denote as Enriched Equity Evaluation Corpus, or EEEC. In this data set, we use the 11 templates of EEC and randomly add a prefix or suffix phrase, which can describe a related place, family member, time, and day, including also the corresponding pronouns to the Gender of the person being discussed. We also create 13 non-informative sentences, and concatenate them before or after the template such that there is a correlation between each label and three of those sentences.16 This is performed so that we have other information that could be valuable for the classifier other than the person’s name and the emotion word. Also, to further prevent memorization, we include emotion words that are ambiguous and can describe multiple mood states. Our enriched data set consists of 33,738 sentences generated by 42 templates that are longer and much more diverse than the templates used in the original EEC. While still synthetic and somewhat unrealistic, our data set has much longer sentences, has more features that are predictive of the label, and is harder for the classifier to memorize.

* *Associated task:* classification (single-label classification)
	* classification of the sentences according to the 'gender', 'race' or 'POMS' (profile of mood states)
* *Domain:* Research / Education
* *Type:* Synthetic
* *Instance count:* 33738 sentences (according to paper)
* *Data types:* String, Numeric
* *Missing values:* ~ (race attributed randomly when missing in the 'gender treatement' splits)
* *Dataset infos and download:* [CausaLM repository](https://github.com/amirfeder/CausaLM) 

Introduced in:
Feder, A., Oved, N., Shalit, U., & Reichart, R. (2021). [Causalm: Causal model explanation through counterfactual language models.](https://direct.mit.edu/coli/article/47/2/333/98518) Computational Linguistics, 47(2), 333-386.

## Personal notes:

The splits provided by the authors of the "CausaLM" article contain pairs of 'factual' and 'counterfactual' examples. For the evaluation of a model's ability to predict 'gender', 'race' or mood state ('POMS'), this notion of pairs is unnecessary. So, for each split in the dataset we collected the unique instances they contain, an instance being either a factual example or a counterfactual example in the data provided. Below are the statistics for the distribution of these unique instances in the different splits and the 'overlaps' between the different splits (i.e. the rate of unique instances that appear in both splits of the complete dataset).

*Gender as a treatment: *
Total number of unique observations: 30,055 unique sentences
Number of unique observations per split and overlap with the other splits:
* train: 25,169 unique sentences (overlap w/ validation: 6,796, w/ test: 8,184)
* validation: 9,505 unique sentences (overlap w/ train: 6,796, w/ test: 3,157)
* test: 11,422 unique sentences (overlap w/ train: 8,184, w/ validation: 3,157)
* overlap between 'train + validation' and 'test': 9,245 (~81% of the train set)
For evaluation, I build a train/test/split that has no overlap between the different splits of the dataset with the following characteristics: 

| Total number of instances| #train    | #validation | #test | Comments                       |
| ------------------------ | --------- | ----------- | ----- | ------------------------------ |
| 30,005 | 25,169 | 2,709 | 2,177 | |
|  | *83.88%* | *9.03%* | *7.26%* | |

POMS distribution in sets (train, validation, test):
* anger: 22.38% - 23.07% - 21.68%
* fear: 23.25% - 23.48% - 25.17%
* joy: 23.42% - 24.10% - 23.89%
* sadness: 23.43% - 23.77% - 22.88%
* neutral: 7.51% - 5.57% - 6.38%

Race_label distribution in sets (train, validation, test):
* African-American: 49.97% - 52.05% - 51.17%
* European: 50.03% - 47.95% - 48.83%

Gender distribution in sets (train, validation, test):
* male: 49.97% - 49.72% - 50.53%
* female: 50.03% - 50.28% - 49.47%


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


