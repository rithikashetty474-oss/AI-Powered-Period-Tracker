"""
LSTM Model for Cycle Prediction
Starter scaffold for cycle prediction using LSTM neural network
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

# Try to import TensorFlow/Keras (optional for starter)
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
    Sequential  # Make sure it's defined
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None  # Define as None if not available
    LSTM = None
    Dense = None
    Dropout = None


class CycleLSTMPredictor:
    """LSTM-based cycle prediction model"""
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'cycle_lstm_model.h5')
        self.sequence_length = 6  # Use last 6 cycles for prediction
    
    def prepare_data(self, cycles: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare cycle data for LSTM training"""
        if len(cycles) < 3:
            return None, None
        
        # Extract cycle lengths
        cycle_lengths = []
        for cycle in cycles:
            try:
                length = int(cycle.get('cycle_length', 28))
                if 20 <= length <= 45:  # Valid cycle length range
                    cycle_lengths.append(length)
            except (ValueError, TypeError):
                continue
        
        if len(cycle_lengths) < 3:
            return None, None
        
        # Create sequences
        X, y = [], []
        for i in range(len(cycle_lengths) - self.sequence_length):
            X.append(cycle_lengths[i:i + self.sequence_length])
            y.append(cycle_lengths[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM: (samples, time_steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, cycles: List[Dict], epochs: int = 100) -> bool:
        """Train the LSTM model on cycle data"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return False
        
        X, y = self.prepare_data(cycles)
        if X is None or len(X) < 3:
            print("Not enough data to train model")
            return False
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Train
        try:
            self.model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)
            self.model.save(self.model_path)
            print(f"Model trained and saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False
    
    def predict_next_cycle(self, cycles: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Predict next cycle start date
        Returns: (predicted_date, confidence)
        """
        if not cycles:
            return None, 0.0
        
        # Rule-based fallback (always available)
        if len(cycles) < 2:
            # Default prediction: 28 days from last cycle
            last_cycle = cycles[-1]
            try:
                last_date = datetime.strptime(last_cycle['start_date'], '%Y-%m-%d')
                predicted = last_date + timedelta(days=28)
                return predicted.strftime('%Y-%m-%d'), 0.5
            except:
                return None, 0.0
        
        # Calculate average cycle length
        cycle_lengths = []
        for cycle in cycles[-6:]:  # Use last 6 cycles
            try:
                length = int(cycle.get('cycle_length', 28))
                if 20 <= length <= 45:
                    cycle_lengths.append(length)
            except:
                continue
        
        if not cycle_lengths:
            cycle_lengths = [28]  # Default
        
        avg_length = sum(cycle_lengths) / len(cycle_lengths)
        
        # Try LSTM prediction if available
        if TENSORFLOW_AVAILABLE and self.model is not None:
            try:
                X, _ = self.prepare_data(cycles)
                if X is not None and len(X) > 0:
                    # Use last sequence
                    last_sequence = X[-1:] if len(X) > 0 else None
                    if last_sequence is not None:
                        prediction = self.model.predict(last_sequence, verbose=0)[0][0]
                        avg_length = max(20, min(45, prediction))  # Clamp to valid range
            except Exception as e:
                print(f"LSTM prediction error, using average: {e}")
        
        # Calculate predicted date
        last_cycle = cycles[-1]
        try:
            last_date = datetime.strptime(last_cycle['start_date'], '%Y-%m-%d')
            predicted = last_date + timedelta(days=int(avg_length))
            
            # Confidence based on data quality
            confidence = min(0.95, 0.5 + (len(cycle_lengths) * 0.1))
            
            return predicted.strftime('%Y-%m-%d'), confidence
        except Exception as e:
            print(f"Error calculating prediction: {e}")
            return None, 0.0


def predict_cycle(cycles: List[Dict]) -> Tuple[Optional[str], float]:
    """Convenience function for cycle prediction"""
    predictor = CycleLSTMPredictor()
    
    # Try to load existing model
    predictor.load_model()
    
    # If we have enough data and no model, train one
    if len(cycles) >= 6 and predictor.model is None and TENSORFLOW_AVAILABLE:
        predictor.train_model(cycles, epochs=50)
    
    return predictor.predict_next_cycle(cycles)

