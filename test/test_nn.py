from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from sklearn import metrics
import numpy as np
import pytest

nn_architecture = [
    {'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},  # Hidden layer
    {'input_dim': 16, 'output_dim': 8, 'activation': 'sigmoid'}  # Output layer with 8 neurons
]

learning_rate = 0.01
random_seed = 8
batch_size = 5
epochs = 100
loss_function = 'binary_cross_entropy'

nn = NeuralNetwork(nn_architecture, learning_rate, random_seed, batch_size, epochs, loss_function)

nn._param_dict = {
    'W1': np.random.rand(64, 16),
    'b1': np.random.rand(16),
    'W2': np.random.rand(16, 8),
    'b2': np.random.rand(8)
}

def test_single_forward():
    # Define test matrices with correct shapes
    W_curr = np.array([[0, 1], [1, -1]])  # Weight matrix (2, 2)
    b_curr = np.array([[0], [-1]])  # Bias vector (2, 1)
    A_prev = np.array([[0.5, 0.5]])  # Previous layer activations (1, 2)
    activation = 'relu'  # ReLU activation function

    # Forward pass
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    # Correct expected results for Z_curr (linear transformation)
    expected_Z_curr = np.array([[0.5, 0], [-0.5, -1]])  # Correct matrix result
    # Expected results for A_curr (after ReLU activation)
    expected_A_curr = np.array([[0.5, 0], [0, 0]])  # After ReLU activation

    # Assertions to check if the output is correct
    assert np.allclose(Z_curr, expected_Z_curr), f"Expected Z_curr: {expected_Z_curr}, but got {Z_curr}"
    assert np.allclose(A_curr, expected_A_curr), f"Expected A_curr: {expected_A_curr}, but got {A_curr}"

    invalid_activation = 'tanh'  # Not supported
    # Expect ValueError for unsupported activation function
    with pytest.raises(ValueError, match="Unsupported activation function"):
        nn._single_forward(W_curr, b_curr, A_prev, invalid_activation)


def test_forward():
    # Define the input (1 sample, 64 features)
    X = np.random.randn(1, 64)  # 1 sample, 64 features (matching the input_dim of the first layer)

    # Perform the forward pass
    output, cache = nn.forward(X)

    # Check if the output shape is correct for a network with an 8-dimensional output
    assert output.shape == (1, 8), f"Expected output shape (1, 8), but got {output.shape}"

    # Check if the cache contains the expected keys and shapes
    assert cache.get("A2").shape == (1, 8), f"Expected A2 shape (1, 8), but got {cache.get('A2').shape}"
    assert cache.get("Z2").shape == (1, 8), f"Expected Z2 shape (1, 8), but got {cache.get('Z2').shape}"

    # Check if the cache contains A1 and Z1 for the hidden layer
    assert cache.get("A1").shape == (1, 16), f"Expected A1 shape (1, 16), but got {cache.get('A1').shape}"
    assert cache.get("Z1").shape == (1, 16), f"Expected Z1 shape (1, 16), but got {cache.get('Z1').shape}"


def test_single_backprop():
    W_curr = np.array([[0, 1], [1, -1]])  # (2, 2)
    b_curr = np.array([[0], [1]])  # (2, 1)
    A_prev = np.array([[1], [-1]])  # (2, 1)
    dA_curr = np.array([[0.5], [0.5]])  # (2, 1)

    Z_curr = np.dot(W_curr, A_prev) + b_curr
    activation = 'relu'

    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

    # Define expected values
    expected_dA_prev = np.array([[0.5, -0.5]])
    expected_dW_curr = np.array([[0], [0]])
    expected_db_curr = np.array([[0], [1]])

    # Test assertions
    assert np.allclose(dA_prev,
                       expected_dA_prev), f"Test failed for dA_prev. Expected {expected_dA_prev}, got {dA_prev}"
    assert np.allclose(dW_curr,
                       expected_dW_curr), f"Test failed for dW_curr. Expected {expected_dW_curr}, got {dW_curr}"
    assert np.allclose(db_curr,
                       expected_db_curr), f"Test failed for db_curr. Expected {expected_db_curr}, got {db_curr}"


def test_predict():
    # Adjust the input size to match the network's expected input features
    X = np.array([[0.5] * 64])

    # Perform prediction
    predict = nn.predict(X)

    # Perform forward pass
    forward_output = nn.forward(X)[0]

    # Check the outputs
    assert np.allclose(predict, forward_output, atol=1e-1), f"Expected: {predict}, but got: {forward_output}"


def test_binary_cross_entropy():
    # Generate random predictions and binary labels
    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    # Compute using scikit-learn cross-entropy loss
    expected_loss = metrics.log_loss(y, y_hat)

    # Compare function output with expected output
    assert np.isclose(nn._binary_cross_entropy(y, y_hat), expected_loss), \
        f"Mismatch!\nExpected: {expected_loss}\nGot: {nn._binary_cross_entropy(y, y_hat)}"


def test_binary_cross_entropy_backprop():
    # Generate random predictions and labels
    y_hat = np.random.uniform(low=0.01, high=0.99, size=(batch_size, 1))  # Avoid extreme values
    y = np.random.randint(low=0, high=2, size=(batch_size, 1))  # Binary labels

    # Compute expected gradient manually
    epsilon = 1e-12
    dA = (-y / (y_hat + epsilon) + (1 - y) / (1 - y_hat + epsilon)) / batch_size

    # Compare with function output
    np.testing.assert_allclose(nn._binary_cross_entropy_backprop(y, y_hat), dA, atol=1e-6)


def test_mean_squared_error():
    # Generate random predictions and ground truth values
    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    # Use scikit-learn mean squared error as the expected loss
    expected_loss = metrics.mean_squared_error(y, y_hat)

    computed_loss = nn._mean_squared_error(y, y_hat)

    # Compare expected and computed values
    assert np.allclose(computed_loss, expected_loss), f"Mismatch: expected {expected_loss}, got {computed_loss}"

def test_mean_squared_error_backprop():
    # Generate random predictions and ground truth values
    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    # Compute expected derivative
    expected_dA = (2 * (y_hat - y)) / batch_size

    computed_dA = nn._mean_squared_error_backprop(y, y_hat)

    # Compare expected and computed values
    assert np.allclose(computed_dA, expected_dA), f"Mismatch: expected {expected_dA}, got {computed_dA}"

def test_sample_seqs():
    # Test case 1: Balanced classes
    seqs = ['ATCG', 'CGTA', 'GCTA', 'TACG']
    labels = [True, False, True, False]
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels, seed=42)

    # Assert that the length of the sampled sequences is the same as the original
    assert len(sampled_seqs) == len(
        seqs), f"Test failed for balanced classes. Expected {len(seqs)}, got {len(sampled_seqs)}"

    # Assert that the sampled sequences contain the same number of positive and negative samples
    assert sampled_labels.count(True) == sampled_labels.count(
        False), f"Test failed for balanced classes. Expected equal counts of True and False labels."

    # Test case 2: All sequences are positive
    seqs = ['ATCG', 'CGTA', 'GCTA', 'TACG']
    labels = [True, True, True, True]

    try:
        sample_seqs(seqs, labels, seed=42)
        print("Test failed for all positive classes: ValueError expected.")
    except ValueError:
        print("Passed all positive classes test.")

    # Test case 3: All sequences are negative
    seqs = ['ATCG', 'CGTA', 'GCTA', 'TACG']
    labels = [False, False, False, False]

    try:
        sample_seqs(seqs, labels, seed=42)
        print("Test failed for all negative classes: ValueError expected.")
    except ValueError:
        print("Passed all negative classes test.")


def test_one_hot_encode_seqs():
    # Test case 1: Simple sequence with known encoding
    seq = ['AGA']
    encoded_array = one_hot_encode_seqs(seq)
    expected_array = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])  # Expected encoding for 'AGA'

    # Assert that output is the expected array
    assert np.array_equal(encoded_array,
                          expected_array), f"Test failed for sequence {seq}. Got {encoded_array}, expected {expected_array}"

    # Test case 2: Empty sequence
    seq = ['']
    encoded_array = one_hot_encode_seqs(seq)
    expected_array = np.array([[]])  # Expected encoding for an empty sequence

    # Assert that output is the expected array
    assert np.array_equal(encoded_array,
                          expected_array), f"Test failed for sequence {seq}. Got {encoded_array}, expected {expected_array}"
