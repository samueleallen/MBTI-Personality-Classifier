##############################################
# Programmer: Sam Allen
# Class: Cpsc 322-01, Fall 2025
# Programming Assignment #6
# 11/14/25
# Description: Program contains various
# functions for data analytics, exploration,
# cleaning, and train/test splitting.
##############################################
import math
import numpy as np
from mypytable import MyPyTable
from myevaluation import accuracy_score, train_test_split, kfold_split, bootstrap_sample

GLOBAL_HEADER = None # Used for decision tree
GLOBAL_ATTRIBUTE_DOMAINS = None

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

def get_feature_subset(X, feature_indices):
    """
    Returns new dataset containing only features specified by feature_indices
    
    Arguments:
        X (list of list of obj): The list of samples
        feature_indices (list of int): The indices of the columns to include in the new subset
        
    Outputs:
        X_subset (list of list of obj): The list of samples with only the selected features
    """
    X_subset = []

    # Iterate through each instance
    for row in X:
        new_row = []
        # Iterate through indices to keep
        for feature_index in feature_indices:
            # Append value from col
            new_row.append(row[feature_index])

        X_subset.append(new_row)
    
    return X_subset

def set_utils_globals(header, attribute_domains):
    """
    Sets global variables required for util functions
    """
    global GLOBAL_HEADER
    global GLOBAL_ATTRIBUTE_DOMAINS

    GLOBAL_HEADER = header
    GLOBAL_ATTRIBUTE_DOMAINS = attribute_domains

def get_attribute_header(X):
    return [f"att{i}" for i in range(len(X[0]))]

def get_attribute_domains(X, header):
    domains = {}

    for i, attr in enumerate(header):
        domains[attr] = sorted(list(set(row[i] for row in X)))

    return domains

def majority_vote(instances):
    # Get class labels
    labels = [instance[-1] for instance in instances]

    # Get frequency of each label
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    # Determine max count of labels
    max_count = max(counts.values())

    # Find majority label(s)
    majority_labels = [label for label, count in counts.items() if count == max_count]

    # On tie, just choose value based on ascending alphabetical order
    majority_labels.sort()

    return majority_labels[0]

def rf_majority_vote(predictions):
    """
    Finds the majority vote from a list of class predictions.

    Arguments:
        predictions(list of obj): A list of class labels predicted by individual trees.

    Outputs:
        obj: The majority class label.
    """
    # Get frequency of each label
    counts = {}
    for label in predictions:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    # Determine max count of labels
    max_count = max(counts.values())

    # Find majority label(s)
    majority_labels = [label for label, count in counts.items() if count == max_count]

    # On tie, just choose value based on ascending alphabetical order
    majority_labels.sort()

    return majority_labels[0]

def compute_entropy(instances):
    """
    Calculates entropy for set of instances
    """
    if not instances:
        return 0.0
    
    # Get class labels
    labels = [instance[-1] for instance in instances]

    # Get frequency of each label
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    total = len(instances)

    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p, 2)
        
    return entropy

def compute_enew(instances, attribute):
    """
    Computes weighted average entrpy for splitting on a specified attribute
    """
    partitions = partition_instances(instances, attribute)
    total_instances = len(instances)
    enew = 0.0

    for partition in partitions.values():
        if len(partition) > 0:
            weight = len(partition) / total_instances
            enew += weight * compute_entropy(partition)
    
    return enew

def select_attribute(instances, attributes):
    # TODO: implement the general Enew algorithm for attribute selection
    # for each available attribute
    #     for each value in the attribute's domain
    #          calculate the entropy for the value's partition
    #     calculate the weighted average for the parition entropies
    # select that attribute with the smallest Enew entropy
    # for now, select an attribute randomly
    if not instances:
        return None
    
    best_attr = None
    min_enew = float('inf')

    for attribute in attributes:
        curr_enew = compute_enew(instances, attribute)
        if curr_enew is None:
            continue
        if curr_enew < min_enew:
            min_enew = curr_enew
            best_attr = attribute
        
    return best_attr

def partition_instances(instances, attribute):
    # this is group by attribute domain (not values of attribute in instances)
    # Returns a dictionary: {attribute_value: [instances]}
    att_index = GLOBAL_HEADER.index(attribute)
    att_domain = GLOBAL_ATTRIBUTE_DOMAINS[attribute]

    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(instances):
    if not instances:
        return False
    # get the class label of the first instance.
    first_class = instances[0][-1]
    for instance in instances:
        # if any label differs, return False immediately.
        if instance[-1] != first_class:
            return False
        
    # if the loop completes without finding differences, return True.
    return True 

def tdidt(current_instances, available_attributes, parent_total=None):
    
    #    Recursively building a decision tree using the TDIDT algorithm.

    #     1. Select the best attribute to split on and create an "Attribute" node.
    #     2. For each value of the selected attribute:
    #         a. Create a "Value" subtree.
    #         b. If all instances in this partition have the same class:
    #             - Append a "Leaf" node
    #         c. If there are no more attributes to select:
    #             - Append a "Leaf" node (handle clash w/majority vote leaf node)
    #         d. If the partition is empty:
    #             - Append a "Leaf" node (backtrack and replace attribute node with majority vote leaf node)
    #         e. Otherwise:
    #             - Recursively build another "Attribute" subtree for this partition
    #               and append it to the "Value" subtree.
    #     3. Append each "Value" subtree to the current "Attribute" node.
    #     4. Return the current tree (nested list structure).
    if parent_total is None:
        parent_total = len(current_instances)

    # Base case 1: Check if all isntances are of same class
    if len(current_instances) > 0 and all_same_class(current_instances):
        label = current_instances[0][-1]
        return ["Leaf", label, len(current_instances), len(current_instances)]
    
    # Base case 2: Check if there are no more attributes to split on
    if len(available_attributes) == 0:
        label = majority_vote(current_instances)
        return ["Leaf", label, len(current_instances), len(current_instances)]
    
    # Else, we recursively iterate
    split_attribute = select_attribute(current_instances, available_attributes)

    if split_attribute is None:
        # IF no attribute to split on, use majority leaf
        label = majority_vote(current_instances)
        return ["Leaf", label, len(attr_partition), parent_total]
    
    new_attributes = available_attributes.copy()
    new_attributes.remove(split_attribute)

    tree = ["Attribute", split_attribute]

    # GRoup data by attr domain
    partitions = partition_instances(current_instances, split_attribute)

    # Now for each partition we repeat until base case
    for attr_value in sorted(partitions.keys()):
        attr_partition = partitions[attr_value]
        value_subtree = ["Value", attr_value]

        # Base case 1: Check if there are no more instances to partition from
        if len(attr_partition) == 0:
            label = majority_vote(current_instances)
            leaf = ["Leaf", label, 0, len(current_instances)]
            value_subtree.append(leaf)

        # Base case 2: Check if all instances of the partition have the same class label
        elif all_same_class(attr_partition):
            label = attr_partition[0][-1]
            leaf = ["Leaf", label, len(attr_partition), len(current_instances)]
            value_subtree.append(leaf)
        
        # Base case 3: Check if there are no more attributes to select
        elif len(new_attributes) == 0:
            label = majority_vote(attr_partition)
            leaf = ["Leaf", label, len(attr_partition), len(current_instances)]
            value_subtree.append(leaf)
    
        # Else, recursively iterate through to create a new subtree
        else:
            subtree = tdidt(attr_partition, new_attributes.copy(), len(attr_partition))
            value_subtree.append(subtree)
        
        tree.append(value_subtree)
    
    return tree
            


def tdidt_predict(tree, instance):
    data_type = tree[0]

    # Base case: if this is a leaf, just return its class label
    if data_type == "Leaf":
        label = tree[1]
        return label
    
    # Recursive case:if we are here, this is an Attribute node
    attribute_name = tree[1]
    attribute_index = GLOBAL_HEADER.index(attribute_name)
    instance_value = instance[attribute_index]

    # Look for the matching value node
    for values in tree[2:]:
        value = values[1]
        subtree = values[2]
        
        if instance_value == value:
            return tdidt_predict(subtree, instance)

def get_rules(tree, attribute_names, class_name, curr_rule, rules):
    """
    Recursively goes through our tree to extract decision rules
    """
    node_type = tree[0]

    # If leaf node, compelete rule
    if node_type == "Leaf":
        label = tree[1]
        rule = f"{curr_rule} THEN {class_name} = {label}"
        rules.append(rule)
        return
    
    # Else, traverse nodes
    if node_type == "Attribute":
        attr_index = GLOBAL_HEADER.index(tree[1])

        if attribute_names is not None and attr_index < len(attribute_names):
            attr_name = attribute_names[attr_index]
        else:
            attr_name = tree[1]
        
        for i in range(2, len(tree)):
            value_node = tree[i]
            attr_value = value_node[1]
            subtree = value_node[2]

            # Build condition for rule
            new_cond = f"{attr_name} == {attr_value}"

            if curr_rule:
                new_rule = f"{curr_rule} AND {new_cond}"
            else:
                new_rule = f"IF {new_cond}"

            # Recursively iterate
            get_rules(subtree, attribute_names, class_name, new_rule, rules)