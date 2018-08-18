import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes=W.shape[1]
  num_samples=X.shape[0]
  for i in range (num_samples):
      score=np.zeros(shape=[num_classes,])
      for j in range(num_classes):
          score[j]=W[:,j].dot(X[i,:])
      score=score-np.max(score)
      loss += -score[y[i]]+np.log(np.sum(np.exp(score)))
      for j in range(num_classes):
          probability=np.exp(score[j])/np.sum(np.exp(score))
          if j==y[i]:
              dW[:,j]+=(-1+probability)*X[i,:]
          else:
              dW[:,j]+=probability*X[i,:]
  loss /= num_samples
  loss += reg*np.sum(W*W)
  dW /= num_samples
  dW += 2*reg*W
          
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
  num_classes=W.shape[1]
  num_samples=X.shape[0]
  scores=np.dot(X,W)
  scores=scores-np.max(scores,axis=1).reshape(-1,1)
  scores_exp=np.exp(scores)
  loss=-scores[range(num_samples),y].reshape(-1,1) \
      +np.log(np.sum(scores_exp,axis=1)).reshape(-1,1)
  loss=np.mean(loss)+reg*np.sum(W*W)
  probability=scores_exp/(np.sum(scores_exp,axis=1).reshape(-1,1))
  probability[range(num_samples),y] += -1
  dW=np.dot(X.T,probability)/num_samples+2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

