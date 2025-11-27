##############################################
# Programmer: Sam Allen
# Class: Cpsc 322-01, Fall 2025
# Programming Assignment #6
# 11/14/25
# Description: Program contains various
# functions for data analytics, exploration,
# cleaning, and train/test splitting.
##############################################

import numpy as np
from .mypytable import MyPyTable
from .myevaluation import accuracy_score, train_test_split, kfold_split, bootstrap_sample

def random_instances(table, num_instances):
    """
    Purpose: Selects and removes 5 random instances from a table
    
    Arguments:
        table: An object of MyPyTable with at least num_instances amount of rows
        num_instances: An integer value representing the number of random instances to be selected
    Outputs:
        random_table: MyPyTable object containing random instances
    """
    np.random.seed(0)

    data = table.data
    n = len(data)

    # Generate random indices without replacing data
    random_indices = np.random.choice(n, size=num_instances, replace=False)

    # Select each random instance
    random_instances_data = [data[i] for i in random_indices]

    # Create new mypytable object
    random_table = MyPyTable(column_names=table.column_names, data=random_instances_data)

    # Sort indices in descending order for deletion
    random_indices_sorted = sorted(random_indices, reverse=True)

    # Remove randomly selected instances from table
    for idx in random_indices_sorted:
        del table.data[idx]

    return random_table

def find_unique_values_and_counts(col):
    """
    Given a column, finds and returns the unique values and count of values.
    
    Arguments:
        col: Numeric, categorical column from mypytable object
    Outputs:
        unique_vals: List of integers
        val_counts: List of integers
    """
    val_tracker = {}
    unique_vals = []
    for val in col:
        # Finds and appends any unique values
        if val not in val_tracker:
            val_tracker[val] = 1
            unique_vals.append(val)
        else:
            val_tracker[val] += 1
    
    val_counts = list(val_tracker.values())
    return unique_vals, val_counts

def create_doe_bins(col):
    """
    Given the mpg column, returns a list representing mpg values converted to DOE ratings
    
    Arguments:
        col: list that represents a numeric, continuous mpg column
    Output:
        ratings: list containing the ratings of each mpg
    """
    ratings = []

    # Filter each mpg into bins
    for mpg in col:
        if mpg <= 13:
             ratings.append(1)
        elif mpg == 14:
            ratings.append(2)
        elif mpg == 15 or mpg == 16:
            ratings.append(3)
        elif 17 <= mpg <= 19:
            ratings.append(4)
        elif 20 <= mpg <= 23:
            ratings.append(5)
        elif 24 <= mpg <= 26:
            ratings.append(6)
        elif 27 <= mpg <= 30:
            ratings.append(7)
        elif 31 <= mpg <= 36:
            ratings.append(8)
        elif 37 <= mpg <= 55:
            ratings.append(9)
        else:
            ratings.append(10)

    return ratings

def create_equal_width_bins(col, num_bins=5):
    """
    Given the mpg column, returns a dict separating each mpg into one of 5 categorical bins. These bins are equally divided by width
    
    Arguments:
        col: Numeric, continuous mpg column from mypytable object
    Output:
        freq: Dictionary containing the frequency of each bin
    """
    freq = {}

    width = (max(col) - min(col)) / num_bins

    # There will be (num_bins + 1) cutoffs
    cutoffs = [min(col) + i*width for i in range(num_bins)]
    cutoffs.append(max(col))

    # Initialize empty dict
    for i in range(num_bins):
        freq[f'[{cutoffs[i]} - {cutoffs[i+1]})'] = 0

    # Sort each value into bins
    for mpg in col:
        if cutoffs[0] <= mpg < cutoffs[1]:
            freq[f'[{cutoffs[0]} - {cutoffs[1]})'] += 1
        elif cutoffs[1] <= mpg < cutoffs[2]:
            freq[f'[{cutoffs[1]} - {cutoffs[2]})'] += 1
        elif cutoffs[2] <= mpg < cutoffs[3]:
            freq[f'[{cutoffs[2]} - {cutoffs[3]})'] += 1
        elif cutoffs[3] <= mpg < cutoffs[4]:
            freq[f'[{cutoffs[3]} - {cutoffs[4]})'] += 1
        else:
            freq[f'[{cutoffs[4]} - {cutoffs[5]})'] += 1
    
    return freq

def normalize(table, col_indices):
    """
    Normalizes the specified columns using min-max norm.
    
    Arguments:
        data: 2D list of data
        col_indices: List of column indices to normalize
    Outputs:
        normalized_data: 2D list of normalized columns
    """
    # First, create deep copy of data
    normalized_data = [row[:] for row in table.data]

    # Normalize each col
    for col_idx in col_indices:
        # Get each value in col
        col_vals = table.get_column(col_idx)
        min_val = min(col_vals)
        max_val = max(col_vals)

        # Normalize column
        if max_val != min_val:
            for i in range(len(normalized_data)):
                normalized_data[i][col_idx] = (table.data[i][col_idx] - min_val) / (max_val - min_val)
        
    # Convert normalized_data to mypytable object
    cols = table.column_names
    normalized_data = MyPyTable(column_names=cols, data=normalized_data)
    return normalized_data

def random_subsample(X, y, k, split, model):
    """
    Creates a train/test split
    
    Arguments:
    X: Features (List of lists)
        y: labels (list)
        k: number of random iterations (int)
        model: Instance of a predictor
    
    Outputs:
        average accuracy
        average error
    """
    accuracies = []
    error_rates = []

    # Loop k times, doing train test split each time
    for i in range(k):
        # Dont feed function a random seed in this case
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        # Train classifier and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model and append results
        curr_acc = accuracy_score(y_test, y_pred)
        curr_error = 1.0 - curr_acc

        accuracies.append(curr_acc)
        error_rates.append(curr_error)

    # Return average accuracy and error
    return round(np.mean(accuracies), 2), round(np.mean(error_rates), 2)

def cross_val_predict(X, y, k, model):
    """
    Calculates the average accuracy and error using k-fold cross validation
    
    Arguments:
        X: Features (List of lists)
        y: labels (list)
        k: number of folds (int)
        model: Instance of a predictor
    
    Outputs:
        average accuracy
        average error
        y_preds_combined: Full list of y_preds for confusion matrix
        y_test_combined: Full list of y_test for confusion matrix
    """
    accuracies = []
    error_rates = []
    y_pred_combined = []
    y_test_combined = []


    folds = kfold_split(X, k, shuffle=True)

    for train_indices, test_indices in folds:
        # Separate into X_Train, X_test, y_train, y_test sets
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]

        # Train classifier and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model and append results
        curr_acc = accuracy_score(y_test, y_pred)
        curr_error = 1.0 - curr_acc

        accuracies.append(curr_acc)
        error_rates.append(curr_error)

        # Append y_test and y_pred for confusion matrix uses
        y_pred_combined.extend(y_pred)
        y_test_combined.extend(y_test)

    # Return average accuracy and error
    return round(np.mean(accuracies), 2), round(np.mean(error_rates), 2), y_pred_combined, y_test_combined

def bootstrap_method(X, y, n_samples, model):
    """
    Calculates the average accuracy and error using bootstrapping
    
    Arguments:
        X: Features (list of lists)
        y: Labels (list)
        k: Number of bootstrap repetitions
        model: The object of the classifier model
    """
    accuracies = []
    error_rates = []
    
    for _ in range(n_samples):
        X_train, X_test, y_train, y_test = bootstrap_sample(X, y, n_samples)

        # Train classifier
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model and append results
        curr_acc = accuracy_score(y_test, y_pred)
        curr_error = 1.0 - curr_acc

        accuracies.append(curr_acc)
        error_rates.append(curr_error)

    # Return average accuracy and error
    return round(np.mean(accuracies), 2), round(np.mean(error_rates), 2)

def clean_cf_matrix(cf_data, labels):
    """
    Cleans up a confusion matrix by adding labels on the far left side and totals and recognition on the right hand side
    
    Arguments:
        cf_data: 2D matrix representing confusion matrix
        labels: 1D list representing labels of confusion matrix.
    
    Outputs:
        new_cf_data: 2D matrix representing cleaned confusion matrix
    """
    data = []

    for i in range(len(cf_data)):
        row = cf_data[i]

        # Calculate total
        row_total = sum(row)

        # Get correctly predicted counts
        correctly_pred = row[i]

        # Calculate recognition
        if row_total > 0:
            recognition = (correctly_pred / row_total) * 100
        else:
            recognition = 0.0
        
        # Create new row with updated data
        new_row = [labels[i]] + row + [row_total, f"{recognition:.2f}%"]
        data.append(new_row)

    return data
