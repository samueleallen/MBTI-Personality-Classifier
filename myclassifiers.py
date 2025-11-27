##############################################
# Description: Program contains various
# classifiers
##############################################
from mysklearn import myutils
import math

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        indices = []

        if self.X_train is None:
            return [], []

        # Iterate through each test instance
        for test_point in X_test:
            all_distances = []

            # Calculate euclidean distance to each training point to determine k nearest neighbors
            for i in range(len(self.X_train)):
                train_point = self.X_train[i]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(test_point, train_point)))
                all_distances.append((dist, i))
        
            # Sort by distance and select k nearest
            all_distances.sort(key=lambda x: x[0])
            k_nearest = all_distances[:self.n_neighbors]

            # Separate distances and indices finally
            distances.append([d for d, i in k_nearest])
            indices.append([idx for j, idx in k_nearest])
        return distances, indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)

        y_preds = []

        # Iterate through each neighbor for each test instance
        for indices in neighbor_indices:
            # Get class label of each neighbor
            neighbor_labels = [self.y_train[i] for i in indices]

            # Calculate most common label
            counts = {}
            for label in neighbor_labels:
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

            most_common = max(counts, key=counts.get)
            y_preds.append(most_common)

        return y_preds

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        counts = {}
        # Count occurrences of each label
        for label in y_train:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        
        if counts:
            # Find label with max count
            self.most_common_label = max(counts, key=counts.get)
        else:
            self.most_common_label = None

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            return []
        
        y_preds = [self.most_common_label] * len(X_test)

        return y_preds


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dictionary): The prior probabilities computed for each
            label in the training set.
        conditionals(dictionary of dictionaries): The conditional probabilities computed for each
            attribute value/label pair in the training set.
        labels(list): The unique labels within the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.conditionals = None
        self.labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        n = len(X_train)

        self.labels = sorted(list(set(y_train)))

        # Group each class label together
        label_groups = {label: [] for label in self.labels}
        for i in range(n):
            label = y_train[i]
            label_groups[label].append(X_train[i])

        # Calculate each prior probability
        for label in self.labels:
            num_class = len(label_groups[label]) # Count each instance in the class
            self.priors[label] = num_class / n
        
        # Now calculate each conditional probability
        self.conditionals = {label: {} for label in self.labels}

        for label in self.labels:
            instances_of_class = label_groups[label]
            num_class = len(instances_of_class)

            # Count each instance of each attribute for this label
            attribute_count = {}
            for i in range(len(X_train[0])):
                attribute_count[i] = {}
                for instance in instances_of_class:
                    attr_val = instance[i]
                    if attr_val not in attribute_count[i]:
                        attribute_count[i][attr_val] = 0
                    attribute_count[i][attr_val] += 1
            
            # Convert each count to probability
            for i in range(len(X_train[0])):
                for attr_val, count in attribute_count[i].items():
                    key = (i, attr_val)
                    self.conditionals[label][key] = count / num_class

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        # Iterate through each instance in test set
        for instance in X_test:
            posterior_prob = {}

            # Iterate through each label
            for label in self.labels:
                prob = self.priors[label]

                # Multiply by coniditional probabilities
                for i in range(len(X_test[0])):
                    attr_val = instance[i]
                    key = (i, attr_val)

                    # Look up conditional prob
                    conditional_prob = self.conditionals[label].get(key)
                    # If it couldnt find key, condition prob is 0
                    if conditional_prob is None:
                        conditional_prob = 0.0
                    prob *= conditional_prob

                # Store posterior prob
                posterior_prob[label] = prob

            # Select label with max posterior prob
            best_label = max(posterior_prob, key=posterior_prob.get)
            y_predicted.append(best_label)

        return y_predicted
