import numpy as np

def im2col_2d(A, kernel_size, stride):
    """sliding window for 2-d array

    Args:
        A ([np.array]): numpy araay of (H, W)
        block_size ([int, int]): kernel size
        stride ([int, int]): stride of sliding window 
    """
    # Parameters
    H, W = A.shape
    row_extent = H - kernel_size[0] + 1
    col_extent = W - kernel_size[1] + 1

    # Get Starting block indices
    start_idx = np.arange(kernel_size[0])[:, None]*W + np.arange(kernel_size[1]) # (kH, kW)
    
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent, step=stride[0])[:, None]*W + np.arange(col_extent, step=stride[1]) # (row_windows, col_windows) 
    # Get all actual indices & index into input array for final output
    index = start_idx.ravel()[:,None] + offset_idx.ravel() # (kH*kW, row_windows*col_windows)
    output = np.take(A, index) # (kH*kW, row_windows*col_windows)
    return output.T

def col2im_2d(A, input_shape, kernel_size, stride):
    """col2im for 2-d array

    Args:
        A ([np.array]): numpy array of (windows_num, window_size)
        input_shape ([int, int]):
        output_shape ([int, int]): H and W of im2col outupt, H*W = windows_num

    """
    input = np.zeros(input_shape)
    # Parameters
    H, W = input_shape
    row_extent = H - kernel_size[0] + 1
    col_extent = W - kernel_size[1] + 1

    # Get Starting block indices
    start_idx = np.arange(kernel_size[0])[:, None]*W + np.arange(kernel_size[1]) # (HW, 1)
    
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent, step=stride[0])[:, None]*W + np.arange(col_extent, step=stride[1]) # (row_extent, col_extent) 
    index = start_idx.ravel()[:,None] + offset_idx.ravel() # (kH*kW, row_windows*col_windows)

    # index (1, row_windows*col_windows, kH*kW), first index is for advanceding indexing of dimension 1
    np.add.at(input.reshape(-1), index.T[None, :, :], A)
    return input


def im2col_2d_test():
    A = np.arange(16).reshape(4, 4)
    input_shape = A.shape
    kernel_size = (3, 1)
    stride = (1, 1)
    windows_view = im2col_2d(A, kernel_size, stride)
    print(A)
    print(windows_view)

    # print(col2im_2d(windows_view, input_shape, kernel_size, stride))

if __name__ == '__main__':
    im2col_2d_test()
