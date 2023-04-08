# Import libraries
import glob
import pandas as pd
import numpy as np
from datasets import Features, Value, ClassLabel, Dataset
from transformers import AutoTokenizer, AutoConfig
import torch


def load_volvo_dataset_config(args):
  try:
    # read data with used column
    data = pd.read_csv("/dbfs/mnt/customs-nlp/Classification-Final-Table-2022-06-14.csv",
                    header=5, dtype={'a':str,'b':str,'c':str},
                    low_memory=False, encoding_errors='backslashreplace')
    data.dropna(axis=0,subset=['Part Description2'],inplace=True)
    # Choose classification type to be used. Can be:
    # 'EXPORT'
    # 'GENERAL'
    CT = 'GENERAL'

    # Filter based on classification type
    data = data.loc[data['Classification Type']==CT]

    # We will use only data with high confidence level for training
    data_conf = data.loc[data['CFL']==1]

    # Keep only required columns
    data_conf = data_conf[['Part Number','Part Description2','Tariff Number']]
    print(data_conf.keys())
  except:
    print("Error reading file")

  # We will build a model that predicts up to a given hierarchical level 
  trf_level = args.trf_level # 2, 4, 6, 8, 10
  # Store shorter labels in new columns. This is not needed if trf_level=10, but we do it for uniformity
  trf_label = 'Tariff Number '+str(trf_level)
  data_conf[trf_label] = data_conf['Tariff Number']//(np.power(10,10-trf_level))
  # Minimum number of samples per label to accept for inclusion of label in the model
  n_samples_min=5
  # String used to identify export files
  # Group by 'Part Description2' in order to check if each group corresponds to unique tarrif number up to level trf_level
  grouped_conf = data_conf.groupby(by='Part Description2')
  tmp=grouped_conf.apply(lambda x: x[trf_label].unique().shape[0]==1)

  # tmp[tmp] contains elements with unique tariff numbers; the index of tmp is Part Description2, use it to keep the appropriate parts in data_conf
  data_unique = data_conf[data_conf['Part Description2'].isin(tmp[tmp].index)]

  # Drop duplicates in Part Description2, to avoid bias towards repeated entries
  data_unique.drop_duplicates(subset=['Part Description2'],inplace=True)

  # Find labels for which there are very few samples
  underrepresented_labels = list(data_unique['Tariff Number'].value_counts()[data_unique['Tariff Number'].value_counts()<n_samples_min].index)
  # Drop samples corresponding to underrepresented labels
  data_unique = data_unique[~data_unique['Tariff Number'].isin(underrepresented_labels)]

  # Convert labels to string (not required?)
  data_unique[trf_label]=data_unique[trf_label].astype(str)
  # Class names are unique label names
  class_names = data_unique[trf_label].unique().tolist()
  # Number of classes
  n_labels = len(class_names)
  print("Number of classes in Volvo datasets: ",n_labels)
  # Define forward and inverse mapping of user supplied labels (string format, classification code) to integer id used internally for classification.
  # Passing these as arguments to AutoConfig allows the classifier to return results using the custom labels.
  label2id = {class_names[i]:i for i in range(len(class_names))}
  id2label = {i:class_names[i] for i in range(len(class_names))}
  config = AutoConfig.from_pretrained("bert-base-cased", label2id=label2id, id2label=id2label, num_labels=n_labels)

  # We will convert the pandas DataFrame to Huggingface Dataset.
  # Define datatypes for the Dataset that we will create. (Here the word features is not used for input variables only)
  features = Features({'Part Number': Value('string'),'Part Description2': Value('string'), 'label': ClassLabel(names=class_names)})

  # Keep only required columns 
  data_unique_red = data_unique[['Part Number','Part Description2',trf_label]]
  data_unique_red.rename(columns={trf_label:'label'},inplace=True) # This naming convention seems to be required by huggingface Trainer. Is there an option controlling it?

  # Convert to Dataset and perform train/test split.
  # preserve_index=False is needed to get rid of non-standard index in pandas DF; not using this option leads to errors. 
  ds = Dataset.from_pandas(data_unique_red, features=features, preserve_index=False)
  tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

  def encode(examples):
      return tokenizer(examples['Part Description2'], truncation=True, padding='max_length')
  # Load the volvo dataset and create data loaders for training, validation and test
  ds = ds.train_test_split(test_size=0.2, seed=38)
  train_dataset, test_dataset = ds['train'], ds['test']
  # reduce test dataset size to original size*0.2
  # test_dataset = test_dataset.select(range(int(len(test_dataset)*0.2)))
  split_set = train_dataset.train_test_split(test_size=0.1, seed=38)
  train_dataset = split_set['train']
  eval_dataset = split_set['test']

  try:
    train_dataset = train_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
  except:
     print("endcode error")
  
  train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
  test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
  eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

  return train_dataset, test_dataset, eval_dataset, config
