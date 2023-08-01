
# Document or sequence classification via Bert

A constantly evolving document. Calculation of baselines for various datasets used in my NLP research and related projects.
No hyperparameter optimization has been carried out to calculate these results, unless otherwise stated.


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


| Method               | max. # tokens    | micro-F1       | macro-F1       | config. id       | Comments                       |
| -------------------- | ---------------- | -------------- | -------------- | ---------------- | ------------------------------ |
| BSC                  | 16               | 72.69          | 19.48          | wikivitals       |                                |
| BSC                  | 128              | 85.74          | 37.36          | wikivitals       |                                |
| BSC                  | 512              | 85.99          | 34.40          | wikivitals       |                                |
| BWA                  | 16               |                |                | wv4_16_BWA       |                                |
| BWA                  | 128              | 85.94          | 36.18          | wv4_128_BWA      |                                |
| BWA                  | 512              | **87.16**      | **37.72**      | wv4_512_BWA      |                                |


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


