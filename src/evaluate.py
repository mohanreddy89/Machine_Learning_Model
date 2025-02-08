import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_data, preprocess_data, split_data, load_model

def evaluate_model(model_path, data_path):
    # Load and preprocess data
    data = load_data(data_path)
    data = preprocess_data(data)
def evaluate_model(model_path, data_path):
    try:
        # Load and preprocess data
        data = load_data(data_path)
        data = preprocess_data(data)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(data, target_column='target')

        # Load the trained model
        model = load_model(model_path)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f'Mean Absolute Error (MAE): {mae:.2f}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')

        return {
            'mae': mae,
            'mse': mse,
            'r2': r2
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    model_path = 'models/model.pkl'
    data_path = 'data/dataset.csv'
    metrics = evaluate_model(model_path, data_path)
    if metrics:
        print("Model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.2f}")def evaluate_model(model_path, data_path):
    try:
        # Load and preprocess data
        data = load_data(data_path)
        data = preprocess_data(data)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(data, target_column='target')

        # Load the trained model
        model = load_model(model_path)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f'Mean Absolute Error (MAE): {mae:.2f}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')

        return {
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    model_path = 'models/model.pkl'
    data_path = 'data/dataset.csv'
    metrics = evaluate_model(model_path, data_path)
    if metrics:
        print("Model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.2f}")import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('model_evaluation.log')
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def evaluate_model(model_path, data_path):
    try:
        # Load and preprocess data
        logger.info('Loading and preprocessing data...')
        data = load_data(data_path)
        data = preprocess_data(data)

        # Split data into training and testing sets
        logger.info('Splitting data into training and testing sets...')
        X_train, X_test, y_train, y_test = split_data(data, target_column='target')

        # Load the trained model
        logger.info('Loading the trained model...')
        model = load_model(model_path)

        # Make predictions
        logger.info('Making predictions...')
        predictions = model.predict(X_test)

        # Evaluate the model
        logger.info('Evaluating the model...')
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        logger.info(f'Mean Absolute Error (MAE): {mae:.2f}')
        logger.info(f'Mean Squared Error (MSE): {mse:.2f}')
        logger.info(f'R-squared (R²): {r2:.2f}')

        return {
            'mae': mae,
            'mse': mse,
            'r2': r2
        }

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    model_path = 'models/model.pkl'
    data_path = 'data/dataset.csv'
    metrics = evaluate_model(model_path, data_path)
    if metrics:
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.2f}")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column='target')

    # Load the trained model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'R-squared (R²): {r2:.2f}')

if __name__ == "__main__":
    evaluate_model('models/model.pkl', 'data/dataset.csv')