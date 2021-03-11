import numpy as np

class GD:
    def __init__(self, max_iter = 500, learning_rate = .01):
        """
        initialize hyperparamters
        Parameters
        -----------------------------------------
        max_iter:int, the maximum number of iteration
                 default value is 500 
        learning_rate:float, the learning rate 
                      default value is .01
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    def fit(self, features_data, target_data, is_plotting=True):
        """
        computes the theta using gradient descent
        Parameters:
        -----------------------------------------
        features_data: array, containning values of the features
        target_data: array, representing the corresponding value for the 
                     target variable
        
        Returns
        ------------------------------------------
        None
        """
        d = len(features_data[0]) # number of features
        N = len(features_data) # number of trainning examples
        counter = 0
        self.theta = np.random.rand(d,1) # initialize theta to randomly
        predictions = features_data @ self.theta 
        while counter < self.max_iter:
            self.theta -= 1/N * self.learning_rate * features_data.T @ (predictions - target_data)
            predictions = features_data @ self.theta 
            error = self.mean_sqaured_error(predictions, target_data)
            counter += 1
        

    def predict(self, data):
        """
        computes predictions for data
        Parameters
        ------------------------------------------
        data: array, representing data to predict
        """
        return data @ self._theta
    
    def mean_sqaured_error(self, predictions, target):
        return np.sum((predictions - target)**2)/len(target)
