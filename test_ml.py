import pytest
import numpy as np
from ml.data import apply_label
from ml.model import (
    train_model,
    compute_model_metrics
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

# Create code for expected output
@pytest.mark.parametrize("inference, expected_output", [
    ([1], ">50K"),
    ([0], "<=50K")
])

def test_apply_labels(inference, expected_output):
    """
    This test checks if the labels return the expected output (1 for >50k or 0 for <=50k)
    """
    # Test returns against expected output
    assert apply_label(inference) == expected_output


# Create sample data to test against
@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=50, n_features=10, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_train_model(sample_data):
    """
    This test checks if the value returned from the train_model function is not a None value
    """
    # Get the sample data for this test
    X_train, X_test, y_train, y_test = sample_data

    # Use the sample data in the train_model definition
    model = train_model(X_train, y_train)

    # Check the models return after inserting the sample data
    assert model is not None


@pytest.fixture
def compute_model_metrics_sample_data():
    # Make sample data to test against the target
    num_samples = 100
    y_true = np.random.randint(2, size = num_samples)
    y_pred = np.random.randint(2, size = num_samples)

    return y_true, y_pred

def test_compute_model_metrics(compute_model_metrics_sample_data):
    """
    This test checks that the output of the metrics definition returns the expected output
    """
    # Get the sample data for this test 
    y_true, y_pred = compute_model_metrics_sample_data

    # Call the function for this test
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Create the expected values
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)

    # Compare the definition scores against the expected scores
    assert precision == expected_precision
    assert recall == expected_recall
    assert fbeta == expected_fbeta
