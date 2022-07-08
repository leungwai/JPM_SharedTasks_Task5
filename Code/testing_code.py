import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from training_code import *
from load_data import initialize_test
from reading_datasets import read_test
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(model_load_location):
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../Datasets/'
    test_data = read_test(dataset_location , split = 'test')

    labels_to_ids = task7_labels_to_ids
    input_data = (test_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time

    tokenizer = AutoTokenizer.from_pretrained(model_load_location)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

    # unshuffled testing data
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    # Getting testing dataloaders
    test_loader = initialize_test(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = False)

    test_ind_f1 = 0
    test_ind_precision = 0
    test_ind_recall = 0

    start = time.time()

    # Run the model with unshuffled testing data
    test_result = testing(model, test_loader, labels_to_ids, device)

    now = time.time()

    print('TIME TO COMPLETE:', (now-start)/60 )
    print()

    return test_result

if __name__ == '__main__':
    n_epochs = 1
    models = ['dccuchile/bert-base-spanish-wwm-uncased', 'xlm-roberta-base', 'bert-base-multilingual-uncased']


    for loop_index in range(5):
        for model_name in models:
            test_print_statement = 'Testing ' + model_name + ' from loop ' + str(loop_index)
            print(test_print_statement)

            model_load_location = '../saved_models_5/' + model_name + '/' + str(loop_index) + '/' 
            
            result_save_location = '../saved_test_result_5/' + model_name + '/' + str(loop_index) + '/'
            
            unformatted_result_save_location = result_save_location + 'unformatted_test_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_test_result.tsv'

            test_result = main(model_load_location)


            print("\n Testing results")
            print(test_result)
            formatted_test_result = test_result.drop(columns=['text'])

            os.makedirs(result_save_location, exist_ok=True)
            test_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_test_result.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Result files saved")

    print("Everything successfully completed")













    
        