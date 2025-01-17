"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

from time import time
import numpy as np
from utils import im2col_2d

def original_conv(image, kernel):
    """ Note: the implementation specified in the homework.
    An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi), dtype=np.float64)

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    for xi in range(Hi):
        for xj in range(Wi):
            patch = padded[xi:xi+Hk, xj:xj+Wk]
            out[xi, xj] = np.sum(patch*kernel)
    ### END YOUR CODE

    return out

def conv(image, kernel):
    """The implementation for fast convolution to accelerate speed for hyper parameter tuning.
    """
    Hk, Wk = kernel.shape
    padded = np.pad(image, [(Hk//2, Hk//2), (Wk//2, Wk//2)], mode='edge')
    sliding_window = im2col_2d(padded, kernel_size=kernel.shape, stride=(1, 1))
    output = sliding_window @ kernel.ravel()
    return output.reshape(image.shape)
# conv = original_conv

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    xx,yy = np.meshgrid(np.arange(size), np.arange(size))
    k = size // 2
    kernel =  1 / (2*np.pi*sigma**2) * np.exp(-((xx-k)**2+(yy-k)**2) / (2*sigma**2)) 
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([-0.5, 0, 0.5]).reshape(1, 3)
    out = conv(img, kernel)

    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([-0.5, 0, 0.5]).reshape(3, 1)
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    gx = partial_x(img)
    gy = partial_y(img)
    G = np.sqrt(gx**2+gy**2)
    theta = (180*np.arctan2(gy, gx)/np.pi+180) % 360
    ### END YOUR CODE
    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    #print(G)
    ### BEGIN YOUR CODE
    theta = theta % 360 # NOTE: to prevent angle floor to 360, which will not processed
    pad_G = np.pad(G, 1)

    index1 = ((theta == 135) | (theta == 315)) & (G >= pad_G[:-2, 2:]) & (G >= pad_G[2:, :-2])
    out[index1] = G[index1]

    index2 = ((theta == 0) | (theta == 180)) & (G >= pad_G[1:-1, :-2]) & (G >= pad_G[1:-1, 2:])
    out[index2] = G[index2]

    index3 = ((theta == 90) | (theta == 270)) & (G >= pad_G[2:, 1:-1]) & (G >= pad_G[:-2, 1:-1])
    out[index3] = G[index3]

    index4 = ((theta == 45) | (theta == 225)) & (G >= pad_G[2:, 2:]) & (G >= pad_G[:-2, :-2])
    out[index4] = G[index4]
    
    ### END YOUR CODE
    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges =  img > high
    weak_edges = (img <= high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    ### YOUR CODE HERE
    # TODO: how to speed up this implementation
    edges = np.copy(strong_edges)
    i = 0
    while i < indices.shape[0]: # (N, 2)
        x, y = indices[i] 
        neighbors = get_neighbors(x, y, H, W)
        for x2, y2 in neighbors:
            if weak_edges[x2, y2] and not edges[x2, y2]:
                edges[x2, y2] = True
                indices = np.concatenate((indices, [[x2, y2]]))
        i += 1
    
    # Speed up version

    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15, verbose=False):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
        verbose: if print time cost for each step
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)

    start_time = time()
    smoothed = conv(img, kernel)
    end_time = time()
    if verbose:
        print('smooth conv time %.2fs' % (end_time - start_time))

    start_time = time()
    G, theta = gradient(smoothed)
    end_time = time()
    if verbose:
        print('gradient time %.2fs' % (end_time - start_time))
    
    # visualization to set double threshold
    # from matplotlib import pyplot as plt
    # plt.figure()
    # print(G)
    # plt.hist(G.ravel(), bins=10)
    # plt.show()

    start_time = time()
    nms = non_maximum_suppression(G, theta)
    end_time = time()
    if verbose:
        print('nms time %.2fs' % (end_time - start_time))

    strong_edges, weak_edges = double_thresholding(nms, high, low)
    start_time = time()
    edge = link_edges(strong_edges, weak_edges)
    end_time = time()
    if verbose:
        print('link edge time %.2fs' % (end_time - start_time))
        print('-'*40)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(xs.shape[0]):
        x, y = xs[i], ys[i]
        for j in range(num_thetas):
            rho = x * cos_t[j] + y * sin_t[j]
            accumulator[int(rho)+diag_len, j] += 1

    ### END YOUR CODE

    return accumulator, rhos, thetas
