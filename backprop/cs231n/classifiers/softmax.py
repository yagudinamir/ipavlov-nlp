import numpy as np
from random import shuffle

def mat_mul_naive(X, Y):
    ans = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(Y.shape[1]):
                ans[i][k] += X[i][j] * Y[j][k]
                   
    return ans
def transpose_naive(X):
    ans = np.zeros((X.shape[1], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ans[j][i] = X[i][j]
    return ans

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
    
    #loss
    A = mat_mul_naive(X, W)
    A_rows = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = np.exp(A[i][j])
            A_rows[i] += A[i][j]
    losses = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        losses[i] = -1. * np.log(A[i][y[i]] / A_rows[i])
    for i in range(losses.shape[0]):
        loss += losses[i]
    loss = loss / losses.shape[0]
    R = 0.0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            R += W[i][j] ** 2
    loss += R * reg
                   
    #grad
    #shape (D, N)
    dA_dW = transpose_naive(X)
    #shape (N, C)
    dL_dA = np.zeros((A.shape[0], A.shape[1]))
    for i in range(dL_dA.shape[0]):
        for j in range(dL_dA.shape[1]):
            k = 0.0
            if j == y[i]:
                k = 1.0
            val = A[i][j] / A_rows[i]
            dL_dA[i][j] = - (k - val) / y.shape[0]
    dW = mat_mul_naive(dA_dW, dL_dA)
    
    dW += W * reg * 2.0
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
    #loss
    A = np.matmul(X, W)
    A = np.exp(A)
    losses = np.array([A[i][y[i]] for i in range(y.shape[0])])
    A_rows = np.sum(A,  axis=1)
    losses = -np.log(losses / A_rows)
    loss = np.sum(losses) / y.shape[0]
    R = np.sum(W ** 2) * reg
    loss += R
    
    #grad
    dA_dW = X.T
    val = -A / A_rows.reshape((-1, 1))
    val[np.arange(val.shape[0]), y] += 1.
    dL_dA = -val / y.shape[0]
    
    dW = np.matmul(dA_dW, dL_dA)
    
    dW += W * reg * 2.0
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
