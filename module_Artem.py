"""This module compresses the given SVD-decomposed matrix
and produces the PNG image out of it
"""

import numpy as np
import matplotlib.image as matim

def matrix_compress_export(U, V, s_list, M, filename):
    """Here we compress the matrices U,s,V from the SVD decomposition
    by leaving only the M largest eigenvalues in the diagonal matrix s
    
    ------------------------
    Input: 
    U, V - unitary N x N matrices from the SVD decomposition of a matrix A,
    so that A = U S V^+;
    s_list is the list of diagonal elements of the matrix S,
    sorted in descending order;
    M - positive integer number. It is the number of the largest eigenvalues in the matrix A, 
    which remain after the compression
    filename - the name of the output PNG file 
    -------------------------
    """
    if U.shape != V.shape or U.shape[0] != U.shape[1] or U.shape[0] != s_list.shape[0]:
        raise ValueError("The size of the input matrices is incorrect!")
    if M > s_list.shape[0] or M < 1:
        raise ValueError("The number of remaining eigenvalues is incorrect!")
    
    matrix_size = U.shape[0]
    
    U_resize = U[:,:M] #np.resize(U,(matrix_size,M))
    s_resize = s_list[:M,:M] #np.resize(s_list,(M,))  
    V_resize = V[:M,:]#np.resize(V,(M,matrix_size))
    
    matrix_compressed = U_resize.dot(np.diag(s_resize)).dot(V_resize)
    matim.imsave(filename, matrix_compressed)