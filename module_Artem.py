"""This module contains two functions: matrix_compress for the compression of the given SVD-decomposed matrix
and the export_image function that produces the PNG image for given matrix
"""

import numpy as np

def matrix_compress(U, V, s_list, M):
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
    -------------------------
    Output:
    The compressed N x N matrix
    """
    if U.shape != V.shape or U.shape[0] != U.shape[1] or U.shape[0] != s_list.shape[0]:
        raise ValueError("The size of the input matrices is incorrect!")
    if M > s_list.shape[0] or M < 1:
        raise ValueError("The number of remaining eigenvalues is incorrect!")
    
    matrix_size = U.shape[0]
    
    U_resize = np.resize(U,(matrix_size,M))
    s_resize = np.resize(s_list,(M,))  
    V_resize = np.resize(V,(M,matrix_size))
      
    return U_resize.dot(np.diag(s_resize)).dot(V_resize)