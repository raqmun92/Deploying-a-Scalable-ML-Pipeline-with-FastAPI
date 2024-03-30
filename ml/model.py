import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Make a base estimator
    base_estimator = DecisionTreeClassifier()
    # Initiate the model
    model = AdaBoostClassifier(base_estimator=base_estimator)

    # Define the hyperparameters for model testing
    hyperparameters = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 1.0],
        'base_estimator__max_depth': [1, 2],
        'base_estimator__min_samples_split': [2, 5],
        'base_estimator__min_samples_leaf': [1, 2]
    }

    # Use a grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator = model, param_grid = hyperparameters, cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best hyperparameters are:", grid_search.best_params_)

    # Define the best model from the results
    best_model = grid_search.best_estimator_

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : model returned in def train_model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)

    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open (path, 'wb') as file:
        pickle.dump(model, file)
    

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open (path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : 
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """

    # Process the data for the slice
    X_slice, y_slice, _, _ = process_data(
        data,
        column_name=column_name,
        slice_value=slice_value,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        training=False,
        lb=lb
    )
    
    # Get the predictions for the slice
    preds = inference(model, X_slice)

    # Get model metrics from slice features and corresponding labels
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
