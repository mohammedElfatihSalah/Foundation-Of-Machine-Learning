class PCA:
  def __init__(self, random_state=10, iterations_power=100, is_it_standardized=False, copy=True):
    self.random_state = random_state
    self.iterations_power = iterations_power
    self.is_it_standardized = is_it_standardized
    self.copy = copy
    self.std = 1
    self.mean = 0
  
  def fit(self, X, k):
    """
    computes the first k principle components
    Paramters
    ----------------------------------------
    X: numpy array, represent the input data

    Returns
    ----------------------------------------
    None
    """
    X_arr = None
    if self.copy:
      X_arr = X.copy()
    else:
      X_arr = X
    
    # center the data
    if not self.is_it_standardized:
      self.mean = np.mean(X, axis=0)
      self.std = np.std(X, axis=0)
      X_arr = X_arr - self.mean
      X_arr /= self.std
      
    
    N = len(X_arr)
    d = len(X_arr[0])
    
    B = np.array([])

    X_arr = X_arr.T
    for i in range(k):
      # eigen vector using power method
      S =  X_arr @ X_arr.T
      b_k = self.get_eigen_vector(S)
      

      if len(B) == 0:
        B = b_k
      else:
        B = np.hstack((B, b_k))

      # update input data
      X_arr = X_arr - B @ B.T @ X_arr

    self.B = B

    # stop when the kth principle component is computed
    pass

  def encode(self, X):
    """
    transform X to the new coordinate system,
    spanned by [b1,b2,...,bm]

    Note:
    ---------
    X must be columns-wise array

    Paramters
    ----------------------------------------
    X: numpy array, the input data with shape d x N where 
       d is the number of features and is the N
       the number of trainning samples.


    Returns
    ----------------------------------------
    Z: numpy array, the transormation with shape M x N
       where N is the number of trainning samples and M is the
       number of principle components
    """
    d = len(X)
    N = len(X[0])
    X_arr = X.copy()
    if not self.is_it_standardized:
       X_arr = (X_arr - self.mean.reshape(d,1))/self.std.reshape(d,1)
    Z = self.B.T @ X_arr
    return Z

  def decode(self, Z):
    """
    transform Z to the original coordinate system,
    spanned by [e1,e2,...,ed]

    Paramters
    ----------------------------------------
    Z: numpy array, the transormation with shape M x N
       where N is the number of trainning samples and M is the
       number of principle components.

    Returns
    ----------------------------------------
    X: numpy array, an approximation to the input data with shape d x N where 
       d is the number of features and is the N
       the number of trainning samples.
    """
    M = len(Z)
    N = len(Z[0])
    d = self.B.shape[0]
    X = self.B @ Z
    if not self.is_it_standardized:
      X = (X * self.std.reshape(d,1)) + self.mean.reshape(d,1)
    return X
    

  def get_eigen_vector(self,X):
    """
    compute the eigen vector of matrix X 
    with the highest eigen value

    Paramters
    -------------------------------------
    X: numpy array, d x d array

    Returns
    -------------------------------------
    b_k: numpy array, d x 1 array which the eigen vector of X
         with the biggest eigen-value
    """
    d = len(X)
    counter = 0
    b_k = np.random.rand(d).reshape(d,1)
    while counter < 1000:
      r = X@b_k
      b_k = r/ np.dot(r.T,r)**.5
      counter+=1
    return b_k
