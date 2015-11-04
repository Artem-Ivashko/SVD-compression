"""This module compresses the given SVD-decomposed matrix
and produces the PNG image out of it
"""

import numpy as np
import matplotlib.image as matim

def matrix_compress_export(UR, sR, VR, UG, sG, VG, UB, sB, VB, M, filename):
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
#    if U.shape != V.shape or U.shape[0] != U.shape[1] or U.shape[0] != s_list.shape[0]:
#        raise ValueError("The size of the input matrices is incorrect!")
#    if M > s_list.shape[0] or M < 1:
#        raise ValueError("The number of remaining eigenvalues is incorrect!")
    
#    matrix_size = U.shape[0]
    
    U_resizeR = UR[:,:M]
    s_resizeR = sR[:M] 
    V_resizeR = VR[:M,:]
    
    U_resizeG = UG[:,:M]
    s_resizeG = sG[:M] 
    V_resizeG = VG[:M,:]
    
    U_resizeB = UB[:,:M]
    s_resizeB = sB[:M] 
    V_resizeB = VB[:M,:]
    
    matrix_compressed = np.zeros((U_resizeR.shape[0],V_resizeR.shape[1],3))
    matrix_compressed[:,:,0] = U_resizeR.dot(np.diag(s_resizeR)).dot(V_resizeR)
    matrix_compressed[:,:,1] = U_resizeG.dot(np.diag(s_resizeG)).dot(V_resizeG)
    matrix_compressed[:,:,2] = U_resizeB.dot(np.diag(s_resizeB)).dot(V_resizeB)
    matim.imsave(filename, matrix_compressed)