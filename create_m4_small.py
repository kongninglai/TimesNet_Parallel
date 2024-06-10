import numpy as np
import pandas as pd
import os

train = True
dataset_file = 'dataset/m4'
info_file = os.path.join(dataset_file, 'M4-info.csv')
train_cache_file = os.path.join(dataset_file, 'training.npz')
test_cache_file = os.path.join(dataset_file, 'test.npz')

group_values = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
m4_info = pd.read_csv(info_file)
values_train = np.load(train_cache_file, allow_pickle=True).tolist()
values_test = np.load(test_cache_file, allow_pickle=True).tolist()

group_size = 500

m4_info_small = pd.DataFrame(columns=m4_info.columns)
values_train_small = []
values_test_small = []
for group_value in group_values:
    group_indices = m4_info[m4_info['SP'] == group_value].index
    if len(group_indices) == 0:
        print(f"Warning: No data for group {group_value}.")
        continue
    
    sampled_indices = group_indices[:group_size] if len(group_indices) >= group_size else group_indices
    m4_info_small = pd.concat([m4_info_small, m4_info.loc[sampled_indices]])
    for index in sampled_indices:
        values_train_small.append(values_train[index])
        values_test_small.append(values_test[index])

values_train_small = np.array(values_train_small)
values_test_small = np.array(values_test_small)

m4_info_small_file = os.path.join(dataset_file, 'm4_info_small.csv')
training_small_file = os.path.join(dataset_file, 'training_small.npy')
test_small_file = os.path.join(dataset_file, 'test_small.npy')

m4_info_small.to_csv(m4_info_small_file, index=False)
np.save(training_small_file, values_train_small)
np.save(test_small_file, values_train_small)
print("New smaller datasets saved.")
