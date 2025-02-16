import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        # Initialize and train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'MSE': mse,
            'R2': r2,
            'Predictions': y_pred
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Convert to DataFrame for better handling
    X = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    
    # Initialize and run model
    model = MLModel()
    X_train, X_test, y_train, y_test = model.prepare_data(X, y)
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    print("Model Performance:")
    print(f"MSE: {results['MSE']:.4f}")
    print(f"R2 Score: {results['R2']:.4f}")