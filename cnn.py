def conv_naive(x, w, b, conv_param):
    """
    A naive Python implementation of a convolution.
    The input consists of an image tensor with height H and
    width W. We convolve each input with a filter F, where the filter
    has height HH and width WW.
    Input:
    - x: Input data of shape (H, W)
    - w: Filter weights of shape (HH, WW)
    - b: Bias for filter
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    Returns an array.
    - out: Output data, of shape (H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    out = None

    H, W = x.shape
    filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions.
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    
    padded_image = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0)
    
    output_h = 1 + (H + 2*pad - filter_height) // stride
    output_w = 1 + (W + 2*pad - filter_width)  // stride

    output = torch.zeros(output_h, output_w)

    i_o = 0
    j_o = 0
    
    for i in range(0, H +2*pad- filter_height + 1,stride):
      j_o=0
      for j in range(0, W+2*pad - filter_width + 1,stride):
        output[i_o][j_o] = (w * padded_image[i : i + filter_height, j : j + filter_width]).sum() + b
        j_o +=1
      i_o+=1
    
    return output
  
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive Python implementation of a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    During padding, 'pad' zeros should be placed symmetrically (i.e., equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    Returns an array.
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    out = None

    N, C, H, W = x.shape

    
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']


    H_1 = 1 + (H + 2 * pad - filter_height) // stride
    W_1 = 1 + (W + 2 * pad - filter_width) // stride

    x = x.clone()
    # x = torch.pad((pad,pad,pad,pad))
    output = torch.zeros((N, num_filters, H_1, W_1))
    
    for k in range(N):
      for i in range(num_filters):
        result = torch.zeros(H_1,W_1)

          output = 
             
        for j in range(C):
          result += conv_naive(x[k,j,:,:],w[i,j,:,:],b[i]/C, conv_param)
          
  
        output[k, i,:,:] = result
       
    # Check dimensions.
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

 
    return output