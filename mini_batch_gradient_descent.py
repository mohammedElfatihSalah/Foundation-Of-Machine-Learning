
import numpy as np

class MBGD:
    "Implementation of batch gradient descent"
    def __init__(self, learning_rate = 1e-5 , batch_size = 16, max_iter = 500, initial_theta_strategy = 'normal', store_cost_while_fitting = False, l2 = 0.0, l1 = 0.0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.initial_theta_strategy = initial_theta_strategy
        self.store_cost_while_fitting = store_cost_while_fitting
    
    def fit(self, X,y):
        """
        Computes the value of theta, 
        which is stored in self.theta,
        Parameters
        ------------------------------------------
        X: numoy array, an array of the features 
           also it should be scaled
        y: numpy array, an array of the target

        Returns
        ------------------------------------------
        None
        """
        X_arr = X # convert the data frame to numpy array
        N = len(X_arr) # number of trainning examples
        y_arr = y.reshape((N,1)) # convert the data frame to numpy array
        d = len(X_arr[0]) # number of features
        assert self.batch_size < N # ensure that batch size is less than N        
        no_epochs = N // self.batch_size
        cost_arr = []
        self._initialize_theta(d)
        counter = 0
        while counter < self.max_iter:
            for i in range(no_epochs-1):
                X_current = X_arr[ i * self.batch_size : (i+1) * self.batch_size ,:]
                y_current = y_arr[ i * self.batch_size : (i+1) * self.batch_size ,:]
                self.theta -= self.learning_rate * X_current.T @ (X_current @ self.theta - y_current)  + l2 * self.theta + l1 * (2 * (self.theta > 0).astype(int) - 1)
            
            if self.store_cost_while_fitting:
                cost_arr.append(self.cost(X_arr, y_arr)[0][0])
            counter += 1
        if self.store_cost_while_fitting:
            return cost_arr

    def predict(self, X):
        """
        Parameters
        ------------------------------------------
        X: numpy array, an array of the input features

        Returns
        ------------------------------------------
        result: numpy array, array of the predicted values
        """
        result = X @ self.theta
        return result
    def cost(self, X, y):
        """
        Parameters:
        ------------------------------------------
        X: numpy array, the input array
        y: numpy array, array of the target values

        Returns
        ------------------------------------------
        result: float, the mean squared error
        """
        N = len(X) # the number of training examples
        predictions = self.predict(X)
        result = np.dot((predictions - y).T, predictions - y)/N
        return result

    def _initialize_theta(self,d):
        """
        Parameters:
        -----------------------------------------
        d: int, the number of features
        """
        if self.initial_theta_strategy == "normal":
            self.theta = np.random.randn(d,1)
        elif self.initial_theta_strategy == 'zeros':
            self.theta = np.zeros((d,1))
        elif self.initial_theta_strategy == 'ones':
            self.theta = np.ones((d,1))
class Test:
    def test_MBGD_fit(self)


