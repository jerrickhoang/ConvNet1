import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W, dtype='float')

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[1]
  num_classes = W.shape[0]
  for i in range(num_trains):
    scores = W.dot(X[:, i])
    scores -= np.max(scores)
    total_probs = 0.0
    for j in range(num_classes):
      total_probs += np.exp(scores[j])
    for j in range(num_classes):
      p_j_given_i = np.exp(scores[j]) / total_probs
      margin = - p_j_given_i * X[:, i].T
      if j == y[i]:
        margin = (1 - p_j_given_i) * X[:, i].T
      dW[j, :] += -margin
    loss += - np.log(np.exp(scores[y[i]]) / total_probs)
  loss /= num_trains
  dW /= num_trains

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Loss.
  scores = W.dot(X)
  scores -= np.max(scores, axis=0)
  correct_scores = scores[y, np.arange(len(y))]
  loss = - np.mean(np.log(np.exp(correct_scores) / np.sum(np.exp(scores), axis=0)))
  loss += 0.5 * reg * np.sum(W * W)

  # Gradient.
  neg_probs = - np.exp(scores) / np.sum(np.exp(scores), axis=0)
  neg_probs[y, np.arange(len(y))] += 1
  dW = neg_probs.dot(X.T)
  dW /= -X.shape[1]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
