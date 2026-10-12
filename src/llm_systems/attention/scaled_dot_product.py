import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape
    dimension_K = d_k[-1]

    matrix_product = Q @ K.T
    squared_dimension_K = np.sqrt(dimension_K)

    A = matrix_product / squared_dimension_K
    solution = (softmax(A)) @ V


    return solution

if __name__ == "__main__":
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    V = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = scaled_dot_product_attention(Q, K, V)
    print("solution")
    print(output)