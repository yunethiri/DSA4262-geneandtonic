
import pandas as pd
import numpy as np

import torch 

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

def sample_with_ratio(group):
    label_1 = group[group['label'] == 1]
    label_0 = group[group['label'] == 0]

    if len(label_0) == 0:
        sample_label_1 = label_1.sample(n=min(len(label_1), 5), random_state=42)
        sample_label_0 = pd.DataFrame()  # No label_0 data
    else:
        max_label_1 = 5 * len(label_0)
        sample_label_1 = label_1.sample(n=min(len(label_1), max_label_1), random_state=42)
        sample_label_0 = label_0.sample(n=min(len(label_0), 2), random_state=42)

    combined_sample = pd.concat([sample_label_1, sample_label_0])
    if len(combined_sample) > 10:
        combined_sample = combined_sample.sample(n=10, random_state=42)
    
    sampled_indices = combined_sample.index
    remaining_data = group.drop(sampled_indices)
    return combined_sample, remaining_data

def split_data(bags, labels):
    bags_train, bags_temp, labels_train, labels_temp = train_test_split(
    bags, labels, test_size=0.4, random_state=42)
    bags_val, bags_test, labels_val, labels_test = train_test_split(
        bags_temp, labels_temp, test_size=0.5, random_state=42)
    
    return bags_train, labels_train, bags_val, labels_val, bags_test, labels_test

def get_pos_neg(bags, labels):
    bags_pos = []
    labels_pos = []
    bags_neg = []
    labels_neg = []

    for bag, label in zip(bags, labels):
        if label == 1:
            bags_pos.append(bag)
            labels_pos.append(label)
        else:
            bags_neg.append(bag)
            labels_neg.append(label)

    return bags_pos, labels_pos, bags_neg, labels_neg

def balance_data(bags_train_pos, labels_train_pos, bags_train_neg, labels_train_neg, desired_ratio):

    num_pos = len(labels_train_pos)
    num_neg = len(labels_train_neg)
    total = num_neg + num_pos

    desired_num_pos = int(total * (desired_ratio[1] / sum(desired_ratio)))
    desired_num_neg = int(total * (desired_ratio[0] / sum(desired_ratio)))

    # Oversample positive samples
    bags_train_pos_resampled, labels_train_pos_resampled = resample(
        bags_train_pos,
        labels_train_pos,
        replace=True,  # Sample with replacement
        n_samples=desired_num_pos,
        random_state=42
    )
    # Undersample neg examples
    bags_train_neg_resampled, labels_train_neg_resampled = resample(
        bags_train_neg,
        labels_train_neg,
        replace=False,  # Sample without replacement
        n_samples=desired_num_neg,
        random_state=42
    )

    bags_train_resampled = bags_train_pos_resampled + bags_train_neg_resampled
    labels_train_resampled = labels_train_pos_resampled + labels_train_neg_resampled

    return bags_train_resampled, labels_train_resampled

def get_model_weights(labels_train, device):
    labels_train_array = np.array(labels_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train_array), y=labels_train_array)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Get positive and negative class weights
    class_weight_0 = class_weights[0]  # Weight for class 0 (negative)
    class_weight_1 = class_weights[1]  # Weight for class 1 (positive)

    return class_weight_0, class_weight_1

