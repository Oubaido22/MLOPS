import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.data_loader import load_iris_data
from src.model import IrisClassifier
# Assuming these imports exist in your project structure
from src.data_loader import load_iris_data
from src.model import IrisClassifier

class TestIrisClassifier:
    def setup_method(self):
        """Setup method that runs before each test"""
        self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data(test_size=0.3, random_state=42)
        self.classifier = IrisClassifier()

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        assert not self.classifier.is_trained
        assert self.classifier.model is not None

    def test_model_training(self):
        """Test model training functionality"""
        self.classifier.train(self.X_train, self.y_train)
        assert self.classifier.is_trained

    def test_model_prediction(self):
        """Test model prediction functionality"""
        self.classifier.train(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test[:5])
        assert len(predictions) == 5
        assert all(isinstance(pred, (np.int32, np.int64, int)) for pred in predictions)

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        self.classifier.train(self.X_train, self.y_train)
        accuracy, report = self.classifier.evaluate(self.X_test, self.y_test)

        assert 0 <= accuracy <= 1
        assert isinstance(report, str)
        assert "precision" in report.lower()

    def test_model_save_load(self, tmp_path):
        """Test model saving and loading"""
        self.classifier.train(self.X_train, self.y_train)

        # Save model
        save_path = tmp_path / "test_model.pkl"
        self.classifier.save_model(str(save_path))
        assert save_path.exists()

        # Load model
        new_classifier = IrisClassifier()
        new_classifier.load_model(str(save_path))
        assert new_classifier.is_trained

        # Verify predictions match
        original_pred = self.classifier.predict(self.X_test[:5])
        loaded_pred = new_classifier.predict(self.X_test[:5])
        assert np.array_equal(original_pred, loaded_pred)

    # ---------------------------------------------------------
    # NEW TESTS ADDED BELOW
    # ---------------------------------------------------------

    def test_input_dimension_mismatch(self):
        """
        Test 1: Data Format Check. 
        Ensure the model raises a ValueError if input features 
        don't match training shape (Iris has 4 features).
        """
        self.classifier.train(self.X_train, self.y_train)
        
        # Create fake data with 3 features instead of 4
        bad_input = np.random.rand(5, 3)
        
        # Depending on your implementation, this usually raises a ValueError 
        # from the underlying sklearn model.
        with pytest.raises(ValueError):
            self.classifier.predict(bad_input)

    def test_prediction_determinism(self):
        """
        Test 2: Function Output Consistency.
        Calling predict twice on the exact same data should result 
        in the exact same output.
        """
        self.classifier.train(self.X_train, self.y_train)
        single_sample = self.X_test[:1]
        
        run_1 = self.classifier.predict(single_sample)
        run_2 = self.classifier.predict(single_sample)
        
        assert np.array_equal(run_1, run_2)

    def test_overfitting_sanity_check(self):
        """
        Test 3: Small Model Sanity Check.
        The model should be able to predict the data it was trained on 
        with very high accuracy. If this fails, the model is not learning.
        """
        self.classifier.train(self.X_train, self.y_train)
        
        # Predict on the training set
        train_predictions = self.classifier.predict(self.X_train)
        
        # Calculate accuracy manually
        accuracy = np.mean(train_predictions == self.y_train)
        
        # Iris is easy; training accuracy should be near perfect (>90%)
        assert accuracy > 0.90

def test_data_loading():
    """Test data loading functionality"""
    X_train, X_test, y_train, y_test = load_iris_data()

    assert X_train.shape[1] == 4  # 4 features
    assert len(np.unique(y_train)) == 3  # 3 classes
    assert len(X_train) + len(X_test) == 150  # Total samples