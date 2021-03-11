class LogisticRegression:
    """
    class implementing losgistic regression from scratch,
    it uses BGD to compute theta
    """
  def __init__(self, learning_rate=1e-1, max_iter= 100):
    """
    Parameters
    --------------------------------------------
    learning_rate: float, the learning rate 
    max_iter: int, the maximum number of iterations
    """
    self.learning_rate = learning_rate
    self.max_iter = max_iter

  def fit(self, X, y):
    """
    Fits the trainning data
    Parameters
    --------------------------------------------
    X: numpy array, represent the input data
    y: numpy array, represent the target variable
    """
    d = len(X[0])
    self._initialize_theta(d)
    counter = 0
    while counter < self.max_iter:
      grad_1,grad_2 = self._gradient(X,y)
      self.theta -= self.learning_rate * grad_1
      self.b  -= self.learning_rate * grad_2
      counter +=1

  def predict(self, X):
    """
    Parameters
    --------------------------------------------
    X: numpy array, represents the input data

    Returns 
    --------------------------------------------
    predictions: numpy array, the predicted values
    """
    z = X @ self.theta + self.b
    sig_predictions = self.sigmoid(z)
    predictions = (sig_predictions > .5).astype(int)  
    return predictions

  def cost(self, predictions, targets):
    return np.sum(predictions.reshape(-1) == targets.reshape(-1))/len(targets)

  def _initialize_theta(self,d):
    """
    initialize theta and bias to random values 
    Parameters
    ------------------------------------------
    d: int, the number of features
    """
    self.theta = np.random.randn(d,1)
    self.b = np.random.rand()

  def _gradient(self, X,y):
    """
    Parameters
    ------------------------------------------
    X: numpy array, the input data
    y: numpy array, the target values

    Returns
    ------------------------------------------
    grad_1: numpy array, the gradient for theta
    grad_2: float, the gradient for the bias
    """
    n = len(y)
    z = X @ self.theta + self.b
    y_predictions = self.sigmoid(z)
    grad_1 = X.T @ (y_predictions.reshape(n,1) - y.reshape(n,1))
    grad_2 = np.sum(y_predictions.reshape(-1) - y.reshape(-1)) 
    return grad_1, grad_2

  def sigmoid(self, z):
    """
    Parameters
    -----------------------------------------
    z: float, int, numpy array

    Returns 
    -----------------------------------------
    the computed sigmoid, if the input is array it's computed elementwise
    """
    return 1./(1 + np.exp(-z))
  
