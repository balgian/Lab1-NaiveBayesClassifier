from collections import defaultdict
import numpy as np


def calculate_class_priors(y):
    # Calculate the prior probabilities for each output class
    class_counts = defaultdict(int)
    for target in y:
        class_counts[target] += 1
    class_priors = {c: count / len(y) for c, count in class_counts.items()}
    return class_priors


def calculate_feature_conditional_probabilities(x, y, alpha):
    # Calculate the conditional probabilities for each feature and each input class
    feature_conditional_probabilities = {}
    for feature_idx in range(x.shape[1]):  # Iterate over each feature from 0 to 3
        feature_counts = defaultdict(lambda: defaultdict(int))
        for input_class in np.unique(x[:, feature_idx]):
            for output_class in np.unique(y):
                feature_counts[input_class][output_class] = np.sum((x[:, feature_idx] == input_class)
                                                                   & (y == output_class))
        feature_conditional_probabilities[feature_idx] = {}
        for input_class in np.unique(x[:, feature_idx]):
            feature_conditional_probabilities[feature_idx][input_class] = {}
            for output_class in np.unique(y):
                # Computing the conditional probability without Laplace smoothing
                feature_conditional_probabilities[feature_idx][input_class][output_class] = (
                        (feature_counts[input_class][output_class]) / (
                        np.sum(list(feature_counts[input_class].values()))))
                # Computing the conditional probability with Laplace smoothing if equal to 0
                if feature_conditional_probabilities[feature_idx][input_class][output_class] == 0:
                    feature_conditional_probabilities[feature_idx][input_class][output_class] = (
                            (feature_counts[input_class][output_class] + alpha) / (
                            np.sum(list(feature_counts[input_class].values())) + alpha * len(
                            np.unique(x[:, feature_idx]))))
    return feature_conditional_probabilities


class DiscreteNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_conditional_probabilities = {}

    def fit(self, x, y):
        # Calculate the prior probabilities for each output class
        self.class_priors = calculate_class_priors(y)
        # Calculate the conditional probabilities for each feature and each input class
        self.feature_conditional_probabilities = calculate_feature_conditional_probabilities(x, y, self.alpha)

    def predict(self, x):
        # Predict the output class for each input sample
        predictions = []
        for input_sample in x:
            output_class_probabilities = {}
            for output_class in self.class_priors:
                output_class_probability = self.class_priors[output_class]
                for feature_idx in range(x.shape[1]):
                    input_class = input_sample[feature_idx]
                    try:
                        output_class_probability *= self.feature_conditional_probabilities[feature_idx][input_class][
                            output_class]
                    except KeyError as e:
                        print(f'The feature {e} is not in the training set. Possible values are '
                              f'{list(self.feature_conditional_probabilities[feature_idx].keys())}')
                        exit(-3)
                output_class_probabilities[output_class] = output_class_probability
            predicted_output_class = max(output_class_probabilities, key=output_class_probabilities.get)
            predictions.append(predicted_output_class)
        return predictions
