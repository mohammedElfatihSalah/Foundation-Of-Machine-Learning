class KMeans1:
  def __init__(self, k = 3, max_iter=100, seed = 10):
    """
    Paramters
    --------------------------------------------
    k: int, the number of centroids
    max_iter: int, the maximum number of iterations
    seed: int, the random seed
    """
    self.k = k
    self.max_iter=max_iter
    self.seed = seed

  def _init_centroids(self, X):
    """
    Paramters
    --------------------------------------------
    X: numpy array, the trainning data from which 
       samples the centroids

    Returns
    --------------------------------------------
    None
    """
    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)
    indices = indices[:self.k]
    self.centroids = X[indices]

  def dist_to_centroids(self,sample):
    dist = np.sum((sample - self.centroids) ** 2 , axis = 1)
    return dist.reshape(-1)
  
  def fit(self, X):
    """
    Parameters
    --------------------------------------------
    X: numpy array, the trainning data that we want
       to cluster
    """
    np.random.seed(self.seed)
    N = len(X) # number of trainning data
    d = len(X[0]) # number of features
    self._init_centroids(X) # initializing centroids
    counter = 0
    while counter < 500:
      # Assignment
      clustered_samples = np.zeros(N)
      for i in range(N):
        sample = X[i,:]
        dist_to_centroids = self.dist_to_centroids(sample)
        winning_centroid = np.argmin(dist_to_centroids)
        clustered_samples[i] = winning_centroid


      
      # Update
      for c in range(len(self.centroids)):
        new_centroid = np.sum(X[clustered_samples == c], axis = 0) / len(X[clustered_samples == c])
        self.centroids[c] = new_centroid
     
      counter+=1
