import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
from django.conf import settings

class GradePredictionModel:
    """Machine Learning Model for Grade Prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_path = os.path.join(settings.BASE_DIR, 'predictor', 'trained_model.pkl')
        self.encoders_path = os.path.join(settings.BASE_DIR, 'predictor', 'label_encoders.pkl')
        
    def prepare_features(self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        
        # Encode categorical variables
        categorical_columns = ['gender']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle new categories during prediction
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # If new category, assign most common encoded value
                        df[f'{col}_encoded'] = 0
        
        # Select features for prediction
        feature_columns = [
            'avg_past_grade', 'total_activities', 'total_hours', 'gender_encoded'
        ]
        
        # Fill missing values
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        self.feature_columns = feature_columns
        return df[feature_columns]
    
    def train_model(self, training_data):
        """Train the prediction model"""
        # Prepare features
        X = self.prepare_features(training_data)
        y = training_data['avg_past_grade']  # Using past grade as target for demonstration
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance - MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Save model and encoders
        self.save_model()
        
        return mae, r2
    
    def predict_grade(self, student_data):
        """Predict grade for a student"""
        if self.model is None:
            self.load_model()
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make prediction
        predicted_grade = self.model.predict(X)[0]
        
        # Calculate confidence (simplified approach)
        confidence = min(95, max(50, 100 - abs(predicted_grade - df.iloc[0].get('avg_past_grade', 75))))
        
        # Determine performance level and risk
        if predicted_grade >= 80:
            performance_level = "Excellent"
            at_risk = False
        elif predicted_grade >= 70:
            performance_level = "Good"
            at_risk = False
        elif predicted_grade >= 60:
            performance_level = "Needs Improvement"
            at_risk = True
        else:
            performance_level = "At Risk"
            at_risk = True
        
        return {
            'predicted_grade': round(predicted_grade, 2),
            'confidence': round(confidence, 2),
            'performance_level': performance_level,
            'at_risk': at_risk
        }
    
    def save_model(self):
        """Save trained model and encoders"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_model(self):
        """Load trained model and encoders"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        if os.path.exists(self.encoders_path):
            with open(self.encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
        
        if self.model is None:
            # Create a simple model for demonstration
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a demo model when no trained model exists"""
        # Create dummy training data
        np.random.seed(42)
        dummy_data = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], 100),
            'avg_past_grade': np.random.normal(75, 15, 100),
            'total_activities': np.random.randint(0, 10, 100),
            'total_hours': np.random.normal(20, 10, 100)
        })
        
        # Ensure positive values
        dummy_data['avg_past_grade'] = np.clip(dummy_data['avg_past_grade'], 40, 100)
        dummy_data['total_hours'] = np.clip(dummy_data['total_hours'], 0, 50)
        
        self.train_model(dummy_data)