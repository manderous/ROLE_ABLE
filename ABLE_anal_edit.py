# Tjy write
import torch
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import scipy.linalg as la


def matrix_similarity_analysis(matrix1, matrix2):
    # Calculating eigenvalues
    eigenvalues1, eigenvectors1 = la.eig(matrix1)
    eigenvalues2, eigenvectors2 = la.eig(matrix2)

    # Compute the main eigenvector (the eigenvector corresponding to the largest eigenvalue)
    main_eigenvector1 = eigenvectors1[:, np.argmax(np.abs(eigenvalues1))]
    main_eigenvector2 = eigenvectors2[:, np.argmax(np.abs(eigenvalues2))]

    # Compute the cosine similarity of the main eigenvectors
    main_eig_similarity = (np.dot(main_eigenvector1, main_eigenvector2) /
                           (np.linalg.norm(main_eigenvector1) * np.linalg.norm(main_eigenvector2)))

    return main_eig_similarity


if __name__=="__main__":
    # ABLE
    # Load the parameters of the three tasks
    A_task_vector = torch.load("MEET_ATT_MAVEN_temporal_cla_requests_yes_no.pth")
    A_task_w_name = list(A_task_vector.keys())[0]
    A_matrix = A_task_vector[A_task_w_name].cpu().numpy()

    B_task_vector = torch.load("MEET_ATT_MAVEN_temporal_ext_requests_yes_no.pth")
    B_task_w_name = list(B_task_vector.keys())[0]
    B_matrix = B_task_vector[B_task_w_name].cpu().numpy()

    C_task_vector = torch.load("MEET_ATT_MAVEN_causal_cla_requests_yes_no.pth")
    C_task_w_name = list(C_task_vector.keys())[0]
    C_matrix = C_task_vector[C_task_w_name].cpu().numpy()

    D_task_vector = torch.load("MEET_ATT_MAVEN_causal_ext_requests_yes_no.pth")
    D_task_w_name = list(D_task_vector.keys())[0]
    D_matrix = D_task_vector[D_task_w_name].cpu().numpy()

    E_task_vector = torch.load("MEET_ATT_MAVEN_subevent_cla_requests_yes_no.pth")
    E_task_w_name = list(E_task_vector.keys())[0]
    E_matrix = E_task_vector[E_task_w_name].cpu().numpy()

    F_task_vector = torch.load("MEET_ATT_MAVEN_subevent_ext_requests_yes_no.pth")
    F_task_w_name = list(F_task_vector.keys())[0]
    F_matrix = F_task_vector[F_task_w_name].cpu().numpy()

    # difference between analogous tasks
    A_B_matrix = A_matrix - B_matrix
    C_D_matrix = C_matrix - D_matrix
    E_F_matrix = E_matrix - F_matrix

    # difference between non-analogous tasks
    A_D_matrix = A_matrix - D_matrix
    C_F_matrix = C_matrix - F_matrix
    E_B_matrix = E_matrix - B_matrix

    A_C_matrix = A_matrix - C_matrix
    C_E_matrix = C_matrix - E_matrix
    E_A_matrix = E_matrix - A_matrix

    # Calculate the similarities of the matrices

    result_right_1 = matrix_similarity_analysis(A_B_matrix, C_D_matrix)
    result_wrong_11 = matrix_similarity_analysis(A_B_matrix, C_F_matrix)
    result_wrong_12 = matrix_similarity_analysis(A_B_matrix, A_D_matrix)
    print(result_right_1)
    print(result_wrong_11)
    print(result_wrong_12)

    result_right_2 = matrix_similarity_analysis(C_D_matrix, E_F_matrix)
    result_wrong_21 = matrix_similarity_analysis(C_D_matrix, E_B_matrix)
    result_wrong_22 = matrix_similarity_analysis(C_D_matrix, C_F_matrix)
    print(result_right_2)
    print(result_wrong_21)
    print(result_wrong_22)

    result_right_3 = matrix_similarity_analysis(E_F_matrix, A_B_matrix)
    result_wrong_31 = matrix_similarity_analysis(E_F_matrix, A_D_matrix)
    result_wrong_32 = matrix_similarity_analysis(E_F_matrix, E_B_matrix)
    print(result_right_3)
    print(result_wrong_31)
    print(result_wrong_32)


