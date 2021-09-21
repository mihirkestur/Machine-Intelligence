import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # distance_matrix has all the distances for each query
        distance_matrix = []
        try:
            # For each query, calculate the distances
            for query in x:
                # Holds distance values for each query
                per_query_distances = []
                # For each sample in dataset calculate distance to query instance
                for sample_record in self.data:
                    # Let distance be 0 initially
                    distance = 0
                    # Accumulate distance for each attribute in the dataset
                    for attribute_query, attribute_sample in zip(query, sample_record):
                        distance += (abs(attribute_query - attribute_sample)) ** (self.p)
                    # Insert distance for each sample
                    per_query_distances.append(distance ** (1/self.p))
                # Insert the distances list into the matrix
                distance_matrix.append(per_query_distances)
        except:
            # Return the matrix as a numpy object
            return np.asarray(distance_matrix)
        # Return the matrix as a numpy object
        return np.asarray(distance_matrix)

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # Obtain distances for each query instance
        distance_neighbors = self.find_distance(x)
        # Answer matrix
        nearest_neighbors = []
        # Holds the distances
        neigh_dists = []
        # Holds the indices
        id_neigh = []
        try:
            # For every query obtain 'k' nearest neighbors and their indices
            for distances in distance_neighbors:
                # Obtain the 'k' nearest neighbor indices
                k_idx = distances.argsort()[:self.k_neigh]
                # Obtain the 'k' nearest neighbor values
                k_dist = [distances[i] for i in k_idx]
                # Append these to the respective lists
                neigh_dists.append(k_dist)
                id_neigh.append(k_idx)
        except:
            # Add the two arrays into the answer matrix
            nearest_neighbors.append(np.asarray(neigh_dists))
            nearest_neighbors.append(np.asarray(id_neigh))
            # Return the answer matrix as a numpy object
            return np.asarray(nearest_neighbors)
        # Add the two arrays into the answer matrix
        nearest_neighbors.append(np.asarray(neigh_dists))
        nearest_neighbors.append(np.asarray(id_neigh))
        # Return the answer matrix as a numpy object
        return np.asarray(nearest_neighbors)
    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # Obtain the 'k' nearest neighbors and their indices
        neighbor_matrix = self.k_neighbours(x)
        # Answer list
        predictions = []
        # Holds the indices of the distances
        id_matrix = neighbor_matrix[1]
        # Dictionary to hold target_class: value (count/weights)
        dictionary = dict()
        try:
            # Obtaining all the target classes
            target_values = set(self.target.flatten())
            # If it is a weighted KNN model
            if(self.weighted == True):
                # Note down the distance matrix to find weights
                dist_matrix = neighbor_matrix[0]
                # For each query compute the weight 
                for i in range(len(id_matrix)):
                    # Initialising the dictionary to 0
                    for target_class in target_values:
                        dictionary[target_class] = 0
                    # Obtaining distance and index for each neighbor
                    for dist, id in zip(dist_matrix[i], id_matrix[i]):
                        # If distance is not 0
                        if(dist != 0):
                            # Calculate weight and accumulate it in the target class
                            dictionary[self.target[int(id)]] += 1/dist
                        else:
                            dictionary[self.target[int(id)]] += 1/(0.000000001)
                    # For each query, find the target class with maximum weight (implies minimum distance)
                    predictions.append(max(dictionary, key = lambda i: dictionary[i]))
            # For vanilla KNN model
            else:
                # For each query check target values
                for query_indices in id_matrix:
                    # Initialising the dictionary to 0
                    for target_class in target_values:
                        dictionary[target_class] = 0
                    # For each index increment count for that target class
                    for id in query_indices:
                        dictionary[self.target[int(id)]] += 1
                    # For each query, find target class with maximum count (majority vote)
                    predictions.append(max(dictionary, key = lambda i: dictionary[i]))
        except:
            return np.asarray(predictions)
        return np.asarray(predictions)
        
    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        try:
            # Let accuracy be 0 initially
            accuracy = 0
            # Note down the number of queries
            total_queries = len(y)
            # Only if there are queries proceed
            if(total_queries != 0 and self.k_neigh <= self.data.shape[0]):
                # Obtain the values predicted by KNN
                prediction_values = self.predict(x)
                # Note the correct predictions (i.e TP or TN)
                correct_predictions = 0
                # Check if predictions has been made for all queries 
                if(len(prediction_values) == total_queries):
                    # Check for each value if its correct or not
                    for i in range(total_queries):
                        # If it is correct increment 
                        if(prediction_values[i] == y[i]):
                            correct_predictions += 1
                    # Compute accuracy
                    accuracy = (correct_predictions/total_queries)*100
            # Return the accuracy
            return accuracy
        except:
            return 0