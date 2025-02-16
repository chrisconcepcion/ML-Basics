import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class MLWorkflow:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Create a relationship: y = 2*x1 + 3*x2 + noise
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        df['target'] = y
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Split features and target
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train_model(self):
        """Train the model"""
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """Evaluate the model"""
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Print results
        print("\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Model Coefficients: {self.model.coef_}")
        print(f"Model Intercept: {self.model.intercept_:.4f}")
        
        return y_pred
    
    def visualize_results(self, y_pred):
        """Visualize the results"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.show()

def main():
    # Initialize workflow
    workflow = MLWorkflow()
    
    # Load data
    print("Loading data...")
    df = workflow.load_data()
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess data
    print("\nPreprocessing data...")
    workflow.preprocess_data(df)
    
    # Train model
    print("\nTraining model...")
    workflow.train_model()
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = workflow.evaluate_model()
    
    # Visualize results
    workflow.visualize_results(y_pred)

if __name__ == "__main__":
    main()