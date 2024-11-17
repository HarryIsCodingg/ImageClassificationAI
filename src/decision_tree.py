import numpy as numpy_library


class DecisionTree:

    def __init__(self, maximum_depth=50):
        self.maximum_depth = maximum_depth

    def fit(self, features, labels):
        self.number_of_classes = len(numpy_library.unique(labels))
        self.tree = self.build_tree(features, labels, depth=0)

    # Tree will start from depth 0 and build until maximum depth
    def build_tree(self, features, labels, depth):
        number_of_samples, number_of_features = features.shape

        # Maximum depth is reached or node is pure
        if depth >= self.maximum_depth or len(numpy_library.unique(labels)) == 1:
            leaf_value = self.find_most_common_label(labels)
            return leaf_value

        best_feature, best_threshold = self.find_best_split(features, labels, number_of_features)
        # Return leaf if no valid split found
        if best_feature is None:
            return self.find_most_common_label(labels)

        left_indices = features[:, best_feature] < best_threshold
        right_indices = features[:, best_feature] >= best_threshold

        left_subtree = self.build_tree(features[left_indices], labels[left_indices], depth + 1)
        right_subtree = self.build_tree(features[right_indices], labels[right_indices], depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    # Returns best feature and threshold with lowest gini index to split the data
    def find_best_split(self, features, labels, number_of_features):
        best_gini_index = float('inf')
        best_feature, best_threshold = None, None

        for feature_index in range(number_of_features):
            thresholds = numpy_library.unique(features[:,  feature_index])
            for threshold in thresholds:
                gini_index = self.calculate_gini_index(features[:, feature_index], labels, threshold)
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    # Returns weighted sum of impurities of the split
    def calculate_gini_index(self, feature_column, labels, threshold):
        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold
        number_of_samples, number_of_left_samples, number_of_right_samples = len(labels), sum(left_indices), sum(right_indices)

        # If left samples are 0 then split right, if right samples are 0 then split left
        if number_of_left_samples == 0:
            return self.calculate_gini(labels[right_indices])
        if number_of_right_samples == 0:
            return self.calculate_gini(labels[left_indices])

        # Calculate Gini impurity for each subset
        gini_left = self.calculate_gini(labels[left_indices])
        gini_right = self.calculate_gini(labels[right_indices])
        return (number_of_left_samples / number_of_samples) * gini_left + (number_of_right_samples / number_of_samples) * gini_right

    # Returns gini impurity for given subset
    def calculate_gini(self, label_subset):
        classes, counts = numpy_library.unique(label_subset, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - numpy_library.sum(probabilities ** 2)

    # Returns label with the majority of the samples in node
    def find_most_common_label(self, labels):
        classes, counts = numpy_library.unique(labels, return_counts=True)
        most_common_class = classes[numpy_library.argmax(counts)]
        return most_common_class

    def traverse_tree(self, feature_vector, node):
        if not isinstance(node, tuple):
            return node
        feature_index, threshold, left_subtree, right_subtree = node
        if feature_vector[feature_index] < threshold:
            return self.traverse_tree(feature_vector, left_subtree)
        else:
            return self.traverse_tree(feature_vector, right_subtree)

    def predict(self, features):
        return numpy_library.array([self.traverse_tree(feature_vector, self.tree) for feature_vector in features])


