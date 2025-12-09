##############################################
# Description: Program contains various
# functions for train/test splitting
##############################################
import numpy as np # use numpy's random number generation
import math
import random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """ 
    if isinstance(test_size, float):
        test_samples = math.ceil(len(X) * test_size) # Calculate num of test samples
    else:
        test_samples = test_size
        
    train_samples = len(X) - test_samples

    indices = list(range(len(X)))
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        
        for i in range(len(X)):
            rand_index = np.random.randint(0, len(X))

            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]

    # Create each train/test set
    X_train = X[:train_samples]
    X_test = X[train_samples:]

    y_train = y[:train_samples]
    y_test = y[train_samples:]
    
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n = len(X)

    indices = list(range(n))

    # Set random state and shuffle
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    fold_size = n // n_splits + 1
    remainder = n % n_splits

    folds = []
    start = 0

    # Create folds
    for i in range(n_splits):
        curr_fold = fold_size
        if i < remainder:
            curr_fold += 1
        # Separate into train/test
        end_of_split = start + curr_fold

        test_indices = indices[start:end_of_split]
        train_indices = indices[:start] + indices[end_of_split:]

        folds.append((train_indices, test_indices))

        start = end_of_split

    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    n = len(X)
    samples_used = n_samples if n_samples is not None else n

    if random_state is not None:
        np.random.seed(random_state)

    sample_indices = np.random.choice(n, size=samples_used, replace=True).tolist()

    # Get 'out of bag' indices
    all_indices = list(range(n))
    samples = sample_indices
    oob_indices = [indice for indice in all_indices if indice not in samples]

    X_sample = [X[i] for i in sample_indices]
    X_out_of_bag = [X[i] for i in oob_indices]

    if y is not None:
        y_sample = [y[i] for i in sample_indices]
        y_out_of_bag = [y[i] for i in oob_indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    label_indexes = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)

    # Initialize n x n matrix
    m = [[0] * n_labels for _ in range(n_labels)]

    # Iterate through predicted values and fill matrix m
    for true_val, pred_val in zip(y_true, y_pred):
        i = label_indexes.get(true_val)
        j = label_indexes.get(pred_val)

        m[i][j] += 1
    
    return m

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    # Calculate correct predictions
    correct = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct += 1

    if not normalize:
        return correct

    # Else, we return the frection
    n = len(y_true)
    if n == 0:
        return 0
    return correct / n

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Determine positive label
    if labels is None:
        unique_labels = sorted(list(set(y_true)))
    else:
        unique_labels = labels

    if pos_label is None:
        # Default to first label in labels
        positive_label = unique_labels[0]
    else:
        positive_label = pos_label
    
    # Set negative label
    negative_label = None
    for label in unique_labels:
        if label != positive_label:
            negative_label = label
            break
    
    # Count true positives and false positives
    tp = fp = 0

    for true_val, pred_val in zip(y_true, y_pred):
        # Predicted positive
        if pred_val == positive_label:
            if true_val == positive_label:
                # True positive
                tp += 1
            elif true_val == negative_label:
                # False positive
                fp += 1

    # Handle case where there are no tp's or fp's
    if (tp + fp) == 0:
        return 0.0
    precision = tp / (tp + fp)

    return precision 

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Determine positive label
    if labels is None:
        unique_labels = sorted(list(set(y_true)))
    else:
        unique_labels = labels

    if pos_label is None:
        # Default to first label in labels
        positive_label = unique_labels[0]
    else:
        positive_label = pos_label
    
    # Set negative label
    negative_label = None
    for label in unique_labels:
        if label != positive_label:
            negative_label = label
            break
    
    # Count true positives and false positives
    tp = fn = 0

    for true_val, pred_val in zip(y_true, y_pred):
        # Get instances that are positive
        if true_val == positive_label:
            if pred_val == positive_label:
                # True positive
                tp += 1
            else:
                # False negative
                fn += 1

    # Handle case where there are no tp's or fp's
    if (tp + fn) == 0:
        return 0.0
    recall = tp / (tp + fn)

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # Calculate precision and recall
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    # Ensure denominator wont be 0
    if (precision + recall) == 0:
        return 0.0
    
    # Calculate f1 score and return
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def stratified_split(X, y, test_size=0.33, random_state=None):
    if random_state:
        np.random.seed(random_state)
    # group indices by label
    label_to_indices = {}
    i = 0
    while i < len(y):
        label = y[i]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
        i += 1

    # shuffle each group
    for label in label_to_indices:
        random.shuffle(label_to_indices[label])

    # compute split per group
    train_indices = []
    test_indices = []

    for label in label_to_indices:
        group = label_to_indices[label]
        n = len(group)
        k = int(n * test_size)

        test_indices.extend(group[:k])
        train_indices.extend(group[k:])

    random.shuffle(test_indices)
    random.shuffle(train_indices)

    # build final splits
    X_train, y_train, X_test, y_test = [], [], [], []

    i = 0
    while i < len(train_indices):
        idx = train_indices[i]
        X_train.append(X[idx])
        y_train.append(y[idx])
        i += 1

    j = 0
    while j < len(test_indices):
        idx = test_indices[j]
        X_test.append(X[idx])
        y_test.append(y[idx])
        j += 1

    return X_train, X_test, y_train, y_test
