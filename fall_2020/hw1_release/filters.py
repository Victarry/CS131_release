"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for xi in range(Hi):
        for xj in range(Wi):
            for ki in range(-Hk//2+1, Hk//2+1):
                for kj in range(-Wk//2+1, Wk//2+1):
                    # ki = ki - Hk // 2
                    # kj = kj - Wk // 2
                    if 0 <= xi - ki < Hi and 0 <= xj - kj < Wi:
                        x = image[xi-ki, xj-kj]
                    else:
                        x = 0
                    y = kernel[ki+Hk//2, kj+Wk//2]
                    out[xi, xj] += x*y
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    a = Hk // 2
    b = Wk // 2
    padded_image = zero_pad(image, a, b)
    kernel = np.flip(kernel, axis=(0, 1))
    ### YOUR CODE HERE
    for xi in range(Hi):
        for xj in range(Wi):
            patch = padded_image[xi:xi+Hk, xj:xj+Wk]
            out[xi, xj] = np.sum(patch*kernel)
    ### END YOUR CODE
    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(g, axis=(0, 1)))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g - g.mean()
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    Hk, Wk = g.shape
    Hi, Wi = f.shape
    a = Hk // 2
    b = Wk // 2
    padded_image = zero_pad(f, a, b)
    out = np.zeros_like(f)

    ### YOUR CODE HERE
    for xi in range(Hi):
        for xj in range(Wi):
            patch = padded_image[xi:xi+Hk, xj:xj+Wk]
            patch = (patch-patch.mean()) / patch.std()

            g = (g-g.mean()) / g.std()
            out[xi, xj] = np.sum(patch*g)
    return out
