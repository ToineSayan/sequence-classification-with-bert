import csv
import json
import random
from collections import Counter

random.seed(42)

path_to_original_data = './EEEC_original_data/'
path_to_output = './EEEC_TC_gender/'
label2gender = {
    '0': 'male', 
    '1': 'female'
    }
label2POMS =  {
    '0': 'neutral', 
    '1': 'anger', 
    '2': 'fear', 
    '3': 'joy', 
    '4': 'sadness'
    }
label2Race = {
    '0': 'European',
    '1': 'African-American'
}
keys_2_keep = ['ID', 'Person', 'Sentence', 'Gender_label', 'Template', 'Race', 'Race_label', 'Emotion_word', 'Emotion', 'POMS_label']


def attribute_race(race, unknown=True): # if no race label, attribute one randomly
    if not race in ['African-American', 'European']:
        if not unknown:
            r = random.choice(['African-American', 'European'])
        else:
            r = 'unknown'
    else: 
        r = race
    return r

print('GENDER TREAMENT - STARTING...')

# # GENDER TREATMENT
# per_split_collection_gender = {}
# per_split_F_CF_couples = {}

# #Extract all sentences (F and CF) per split (and remove duplicates) and the couples (F, CF)
# for split in ['train', 'dev', 'test']:
#     data_file = path_to_original_data + f'gender_{split}.csv'
#     output_file = path_to_output + f'EEEC_gender_{split}.json'

#     data_collection = []
#     list_of_F_CF_couples = []

#     with open(data_file, 'r') as f:
#         data_reader = csv.DictReader(f, delimiter=',')
#         for row in data_reader:

#             keys_F = ['ID_F', 'Person_F', 'Sentence_F', 'Gender_F_label', 'Template', 'Race', 'Race_label', 'Emotion_word', 'Emotion', 'POMS_label']
#             observation_F = {i:row[i] for i in keys_F}
#             observation_F['ID'] = observation_F['ID_F']
#             observation_F['Person'] = observation_F['Person_F']
#             observation_F['Sentence'] = [observation_F['Sentence_F']]
#             observation_F['Gender_label'] = [label2gender[observation_F['Gender_F_label']]]
#             observation_F['Race_label'] = [attribute_race(observation_F['Race'])]
#             observation_F['POMS_label'] = [label2POMS[observation_F['POMS_label']]]
#             all_keys = observation_F.keys()
#             observation_F = dict([(key, val) for key, val in observation_F.items() if key in keys_2_keep])
#             data_collection.append(observation_F)

#             keys_CF = ['ID_CF', 'Person_CF', 'Sentence_CF', 'Gender_CF_label', 'Template', 'Race', 'Race_label', 'Emotion_word', 'Emotion', 'POMS_label']
#             observation_CF = {i:row[i] for i in keys_CF}
#             observation_CF['ID'] = observation_CF['ID_CF']
#             observation_CF['Person'] = observation_CF['Person_CF']
#             observation_CF['Sentence'] = [observation_CF['Sentence_CF']]
#             observation_CF['Gender_label'] = [label2gender[observation_CF['Gender_CF_label']]]
#             observation_CF['Race_label'] = [attribute_race(observation_CF['Race'])]
#             observation_CF['POMS_label'] = [label2POMS[observation_CF['POMS_label']]]
#             all_keys = observation_CF.keys()
#             observation_CF = dict([(key, val) for key, val in observation_CF.items() if key in keys_2_keep])
#             data_collection.append(observation_CF)

#             list_of_F_CF_couples.append((observation_F['Sentence'][0], observation_CF['Sentence'][0], observation_F['Template']))

#         # there are 'doubles' in the different splits. Let's remove the doubles
#         no_double = {data_collection[i]['Sentence'][0]:data_collection[i] for i in range(len(data_collection))}
#         per_split_collection_gender[split] = no_double
#         per_split_F_CF_couples[split] = list_of_F_CF_couples

# # Handle the F, CF couples and save them
# full_list_of_couples = []
# for split in per_split_F_CF_couples.keys():
#     full_list_of_couples += per_split_F_CF_couples[split]
# full_list_of_couples = [(a,b,c) if a>b else (b,a,c) for a,b,c in full_list_of_couples]
# full_list_of_couples = [t for t in (set(tuple(i) for i in full_list_of_couples))]
# with open(path_to_output + f'EEEC_gender_F_CF_couples.json', 'w') as o:
#     for f, cf, template in full_list_of_couples:
#         d = {"F":f, "CF": cf, "Template": template}
#         o.write(json.dumps(d) + '\n')


# # Evaluate the overlap between splits (= number of sentences present in 2 different split)
# # Also determine the total number of unique sentences
# for split in per_split_collection_gender.keys():
#     print(split, len(per_split_collection_gender[split].items()))
#     for inter_split in per_split_collection_gender.keys():
#         tmp = set(per_split_collection_gender[split].keys()).intersection(set(per_split_collection_gender[inter_split].keys()))
#         print(f"overlap {split} {inter_split}: {len(tmp)}")
# tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
# tmp = tmp.intersection(set(per_split_collection_gender["test"].keys()))
# print(f"Overlap 'train'+'dev' and 'test': {len(tmp)}")
# tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
# tmp = tmp.union(set(per_split_collection_gender["test"].keys()))
# print(f"Total number of unique sentences: {len(tmp)}")
# print('\n')

# # Build train, validation, and test splits with no overlap between splits (each unique sentence only appears in one unique split)
# data_collection_train = [val for key,val in per_split_collection_gender["train"].items()]
# tmp = set(per_split_collection_gender['dev'].keys()).difference(set(per_split_collection_gender['train'].keys()))
# data_collection_dev = [val for key,val in per_split_collection_gender["dev"].items() if key in tmp]
# tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
# tmp = set(per_split_collection_gender['test'].keys()).difference(tmp)
# data_collection_test = [val for key,val in per_split_collection_gender["test"].items() if key in tmp]


# # Original data is incmplete, let's attribute a random race value when missing (if the sentence does not contain any race information for example)
# # Around 30% of data in each split is missing a race value
# def assign_random_race(data_collection):
#     output = []
#     for v in data_collection:
#         v_ = v
#         v_['Race_label'] = [attribute_race(v['Race_label'][0], unknown=False)]
#         output.append(v_)
#     return(output)

# data_collection_train = assign_random_race(data_collection_train)
# data_collection_dev = assign_random_race(data_collection_dev)
# data_collection_test = assign_random_race(data_collection_test)

# def distributions_of_labels(label_col):
#     count_label_train = Counter([data_collection_train[i][label_col][0] for i in range(len(data_collection_train))])
#     count_label_dev = Counter([data_collection_dev[i][label_col][0] for i in range(len(data_collection_dev))])
#     count_label_test = Counter([data_collection_test[i][label_col][0] for i in range(len(data_collection_test))])
#     print(f"{label_col} distribution in sets (train, validation, test):")
#     [print(f"{i}: {100*count_label_train[i]/len(data_collection_train):.2f}% - {100*count_label_dev[i]/len(data_collection_dev):.2f}% - {100*count_label_test[i]/len(data_collection_test):.2f}%") for i in count_label_test.keys()]

# distributions_of_labels("POMS_label")
# print('\n')
# distributions_of_labels("Race_label")
# print('\n')
# distributions_of_labels("Gender_label")
# print('\n')

# # Save the splits
# collections = [data_collection_train, data_collection_dev, data_collection_test]
# splits = ['train', 'val', 'test']
# for i in range(3):
#     with open(path_to_output + f'EEEC_gender_{splits[i]}.json', 'w') as o:
#         for d in collections[i]:
#             o.write(json.dumps(d) + '\n')

print('RACE TREAMENT - STARTING...')

# RACE TREATMENT

path_to_original_data = './EEEC_original_data/'
path_to_output = './EEEC_TC_race/'

per_split_collection_gender = {}
per_split_F_CF_couples = {}

#Extract all sentences (F and CF) per split (and remove duplicates) and the couples (F, CF)
for split in ['train', 'dev', 'test']:
    data_file = path_to_original_data + f'race_{split}.csv'
    output_file = path_to_output + f'EEEC_race_{split}.json'

    data_collection = []
    list_of_F_CF_couples = []

    with open(data_file, 'r') as f:
        data_reader = csv.DictReader(f, delimiter=',')
        for row in data_reader:

            keys_F = ['ID_F', 'Person_F', 'Sentence_F', 'Race_F_label', 'Template', 'Gender', 'Gender_label', 'Emotion_word', 'Emotion', 'POMS_label']
            observation_F = {i:row[i] for i in keys_F}
            observation_F['ID'] = observation_F['ID_F']
            observation_F['Person'] = observation_F['Person_F']
            observation_F['Sentence'] = [observation_F['Sentence_F']]
            observation_F['Gender_label'] = [observation_F['Gender']]
            observation_F['Race_label'] = [label2Race[observation_F['Race_F_label']]]
            observation_F['POMS_label'] = [label2POMS[observation_F['POMS_label']]]
            all_keys = observation_F.keys()
            observation_F = dict([(key, val) for key, val in observation_F.items() if key in keys_2_keep])
            data_collection.append(observation_F)

            keys_CF = ['ID_CF', 'Person_CF', 'Sentence_CF', 'Race_CF_label', 'Template', 'Gender', 'Gender_label', 'Emotion_word', 'Emotion', 'POMS_label']
            observation_CF = {i:row[i] for i in keys_CF}
            observation_CF['ID'] = observation_CF['ID_CF']
            observation_CF['Person'] = observation_CF['Person_CF']
            observation_CF['Sentence'] = [observation_CF['Sentence_CF']]
            observation_CF['Gender_label'] = [observation_CF['Gender']]
            observation_CF['Race_label'] = [label2Race[observation_CF['Race_CF_label']]]
            observation_CF['POMS_label'] = [label2POMS[observation_CF['POMS_label']]]
            all_keys = observation_CF.keys()
            observation_CF = dict([(key, val) for key, val in observation_CF.items() if key in keys_2_keep])
            data_collection.append(observation_CF)

            list_of_F_CF_couples.append((observation_F['Sentence'][0], observation_CF['Sentence'][0], observation_F['Template']))

        # there are 'doubles' in the different splits. Let's remove the doubles
        no_double = {data_collection[i]['Sentence'][0]:data_collection[i] for i in range(len(data_collection))}
        per_split_collection_gender[split] = no_double
        per_split_F_CF_couples[split] = list_of_F_CF_couples

# Handle the F, CF couples and save them
full_list_of_couples = []
for split in per_split_F_CF_couples.keys():
    full_list_of_couples += per_split_F_CF_couples[split]
full_list_of_couples = [(a,b,c) if a>b else (b,a,c) for a,b,c in full_list_of_couples]
full_list_of_couples = [t for t in (set(tuple(i) for i in full_list_of_couples))]
with open(path_to_output + f'EEEC_gender_F_CF_couples.json', 'w') as o:
    for f, cf, template in full_list_of_couples:
        d = {"F":f, "CF": cf, "Template": template}
        o.write(json.dumps(d) + '\n')


# Evaluate the overlap between splits (= number of sentences present in 2 different split)
# Also determine the total number of unique sentences
for split in per_split_collection_gender.keys():
    print(split, len(per_split_collection_gender[split].items()))
    for inter_split in per_split_collection_gender.keys():
        tmp = set(per_split_collection_gender[split].keys()).intersection(set(per_split_collection_gender[inter_split].keys()))
        print(f"overlap {split} {inter_split}: {len(tmp)}")
tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
tmp = tmp.intersection(set(per_split_collection_gender["test"].keys()))
print(f"Overlap 'train'+'dev' and 'test': {len(tmp)}")
tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
tmp = tmp.union(set(per_split_collection_gender["test"].keys()))
print(f"Total number of unique sentences: {len(tmp)}")
print('\n')

# Build train, validation, and test splits with no overlap between splits (each unique sentence only appears in one unique split)
data_collection_train = [val for key,val in per_split_collection_gender["train"].items()]
tmp = set(per_split_collection_gender['dev'].keys()).difference(set(per_split_collection_gender['train'].keys()))
data_collection_dev = [val for key,val in per_split_collection_gender["dev"].items() if key in tmp]
tmp = set(per_split_collection_gender["train"].keys()).union(set(per_split_collection_gender["dev"].keys()))
tmp = set(per_split_collection_gender['test'].keys()).difference(tmp)
data_collection_test = [val for key,val in per_split_collection_gender["test"].items() if key in tmp]


# # Original data is incmplete, let's attribute a random race value when missing (if the sentence does not contain any race information for example)
# # Around 30% of data in each split is missing a race value
# def assign_random_race(data_collection):
#     output = []
#     for v in data_collection:
#         v_ = v
#         v_['Race_label'] = [attribute_race(v['Race_label'][0], unknown=False)]
#         output.append(v_)
#     return(output)

# data_collection_train = assign_random_race(data_collection_train)
# data_collection_dev = assign_random_race(data_collection_dev)
# data_collection_test = assign_random_race(data_collection_test)

def distributions_of_labels(label_col):
    count_label_train = Counter([data_collection_train[i][label_col][0] for i in range(len(data_collection_train))])
    count_label_dev = Counter([data_collection_dev[i][label_col][0] for i in range(len(data_collection_dev))])
    count_label_test = Counter([data_collection_test[i][label_col][0] for i in range(len(data_collection_test))])
    print(f"{label_col} distribution in sets (train, validation, test):")
    [print(f"{i}: {100*count_label_train[i]/len(data_collection_train):.2f}% - {100*count_label_dev[i]/len(data_collection_dev):.2f}% - {100*count_label_test[i]/len(data_collection_test):.2f}%") for i in count_label_test.keys()]

distributions_of_labels("POMS_label")
print('\n')
distributions_of_labels("Race_label")
print('\n')
distributions_of_labels("Gender_label")
print('\n')

# Save the splits
collections = [data_collection_train, data_collection_dev, data_collection_test]
splits = ['train', 'val', 'test']
for i in range(3):
    with open(path_to_output + f'EEEC_race_{splits[i]}.json', 'w') as o:
        for d in collections[i]:
            o.write(json.dumps(d) + '\n')

