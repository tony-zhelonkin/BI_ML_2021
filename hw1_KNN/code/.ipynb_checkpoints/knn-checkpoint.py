import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):      
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        two loops distances
        """
        num_test = X_test.size
        num_train = self.train__X.size
        distance_mtx = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_range):
                distance_mtx[i,j]= np.sum(abs(i - j))
        return distance_mtx
        
        #pass


    def compute_distances_one_loop(self, X):
        """
        one loop distances
        """
        num_test = X_test.size
        num_train = self.train__X.size
        distance_mtx = np.zeros((num_test, num_train))
        for i in range(X.size):
            a = abs(self.train_X - X[i])
        return distance_mtx

        #pass


    def compute_distances_no_loops(self, X):
        """
        no loops distances
        """
        distance_mtx = np.sum(
            np.abs(
                X[:, None] - self.train_X
            ), axis = -1
        )
        return distance_mtx
        #pass


    def predict_labels_binary(self, distances):
        """
        binary class prediction
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range(n_test):
            # some magic code
            for j in range(2):
                # some more magic
                return prediction
            #pass


    def predict_labels_multiclass(self, distances):
        """
        multiclass prediction
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        n_classes = np.unique(self.train_y).size
        for i in range(n_test):
            # some magic
            for j in range(n_classes):
            # some more magic
            #pass
