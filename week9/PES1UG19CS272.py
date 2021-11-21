import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        # Holds values of clusters for each sample
        c_values = []
        # Note down number of samples
        num_samples = len(data)
        # Find the closest cluster for each sample
        for i in range(num_samples):
            # Let cluster be -1 initially
            curr_c = -1
            # For every cluster calculate distance to current sample
            for j in range(self.n_cluster):
                # Init distance as 0
                dist = 0
                # Using euclidean distance and summing it
                for f in range(len(data[i])):
                    dist += (data[i][f]-self.centroids[j][f])**2
                # If cluster is -1 then assign both curr_c and min_dist to be of current values, else if calculated distance is less than the min_dist re-assign 
                if(curr_c == -1 or dist < min_dist):
                    curr_c = j
                    min_dist = dist
            # Append values to final array
            c_values.append(curr_c)
        return c_values

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
            cluster_assign: Cluster Assignment
        Change self.centroids
        """
        # Holds new centroids
        new_centroids = []
        # Note down num of features
        num_features = len(data[0])
        # Note down num of samples
        num_samples = len(data)
        # Calculate the new centroid for each cluster
        for k in range(self.n_cluster):
            # Has the initial values of features for each cluster
            per_cluster_c = [0]*num_features
            # Has the number of points for each cluster
            per_cluster_points = 0
            # Check for each sample 
            for sample in range(num_samples):
                # If sample belongs to current cluster k
                if(k == cluster_assgn[sample]):
                    per_cluster_points += 1
                    per_cluster_c = [per_cluster_c[f]+data[sample][f] for f in range(num_features)]
            new_centroids.append([i/per_cluster_points for i in per_cluster_c])

        self.centroids = new_centroids

    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        # Init j = 0
        objective_fun = 0
        # Note sample size and feature size
        num_features = len(data[0])
        num_samples = len(data)
        # For all samples calculate distance w.r.t to its respective centroid and sum it 
        for sample in range(num_samples):
            for f in range(num_features):
                objective_fun += (data[sample][f] - self.centroids[cluster_assign[sample]][f])**2
        # return j
        return objective_fun