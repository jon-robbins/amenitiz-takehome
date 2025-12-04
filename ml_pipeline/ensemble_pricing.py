"""
Ensemble Pricing Model.

Combines predictions from multiple approaches:
1. KNN similarity pricing (best individual: 21% MAPE)
2. Relative pricing (26.8% MAPE)
3. Competitor feature model (24.3% MAPE)

Ensemble methods:
- Simple average
- Weighted average (by inverse MAPE)
- Stacking (meta-learner)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, List
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from lib.db import init_db
from ml_pipeline.features import engineer_all_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES
from ml_pipeline.knn_pricing import KNNPricer, load_hotel_prices
from ml_pipeline.competitor_features import add_competitor_features, add_competitor_features_from_reference


MODEL_DIR = Path(__file__).parent.parent / 'outputs' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data and prepares all features needed for ensemble.
    """
    con = init_db()
    
    sql_file = Path(__file__).parent.parent / 'notebooks/eda/05_elasticity/QUERY_LOAD_HOTEL_MONTH_DATA.sql'
    query = sql_file.read_text(encoding='utf-8')
    df = con.execute(query).fetchdf()
    
    df = engineer_all_features(df)
    
    # Core market filter
    df = df[(df['avg_adr'] >= 50) & (df['avg_adr'] <= 250)]
    df = df.dropna(subset=['avg_adr', 'latitude', 'longitude'])
    
    # Temporal split
    df['month'] = pd.to_datetime(df['month'])
    train_df = df[df['month'] < '2024-06-01'].copy()
    test_df = df[df['month'] >= '2024-06-01'].copy()
    
    return train_df, test_df


class EnsemblePricer:
    """
    Ensemble model that combines KNN, relative pricing, and feature-based predictions.
    """
    
    def __init__(self):
        self.knn_pricer = None
        self.feature_model = None
        self.feature_encoders = None
        self.feature_scaler = None
        self.peer_stats = None
        self.ensemble_weights = None
        self.meta_model = None
        self.train_df = None  # Keep for reference
    
    def fit(self, train_df: pd.DataFrame) -> 'EnsemblePricer':
        """
        Fits all component models.
        """
        print("=" * 80)
        print("TRAINING ENSEMBLE MODEL")
        print("=" * 80)
        
        self.train_df = train_df.copy()
        
        # 1. Fit KNN pricer
        print("\n1. Fitting KNN pricer...")
        self.knn_pricer = KNNPricer(k=20, max_distance_km=50)
        self.knn_pricer.fit(train_df)
        
        # 2. Calculate peer statistics
        print("\n2. Calculating peer statistics...")
        self._fit_peer_stats(train_df)
        
        # 3. Add competitor features
        print("\n3. Adding competitor features...")
        train_df = add_competitor_features(train_df, radius_km=10.0)
        self.train_df = train_df
        
        # 4. Fit feature-based model
        print("\n4. Fitting feature-based model...")
        self._fit_feature_model(train_df)
        
        print("\n✓ Ensemble fitted")
        return self
    
    def _fit_peer_stats(self, df: pd.DataFrame) -> None:
        """Fits peer group statistics."""
        peer_cols = ['city_standardized', 'room_type', 'capacity_quartile']
        
        df_copy = df.copy()
        for col in peer_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).fillna('unknown')
            else:
                df_copy[col] = 'unknown'
        
        self.peer_stats = df_copy.groupby(peer_cols, observed=True).agg({
            'avg_adr': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        self.peer_stats.columns = peer_cols + ['peer_mean', 'peer_median', 'peer_std', 'peer_count']
    
    def _fit_feature_model(self, df: pd.DataFrame) -> None:
        """Fits the feature-based model."""
        self.feature_encoders = {}
        
        df_encoded = df.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.feature_encoders[col] = le
        
        # Select features
        competitor_features = ['competitor_median', 'competitor_mean', 'competitor_std', 'competitor_count']
        feature_cols = []
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES + competitor_features:
            if col in df_encoded.columns:
                feature_cols.append(col)
        
        X = df_encoded[feature_cols].fillna(0)
        y = np.log(df['avg_adr'])
        
        # Scale numeric
        numeric_cols = [c for c in NUMERIC_FEATURES + competitor_features if c in X.columns]
        self.feature_scaler = StandardScaler()
        X[numeric_cols] = self.feature_scaler.fit_transform(X[numeric_cols])
        
        self.feature_cols = feature_cols
        
        # Train model
        self.feature_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.feature_model.fit(X, y)
    
    def _get_knn_prediction(self, row: pd.Series) -> float:
        """Gets KNN prediction for a single row."""
        row_df = pd.DataFrame([row])
        pred = self.knn_pricer.predict_single(row_df, exclude_hotel_id=row.get('hotel_id'))
        return pred.predicted_price
    
    def _get_peer_prediction(self, row: pd.Series) -> float:
        """Gets peer group mean prediction."""
        peer_cols = ['city_standardized', 'room_type', 'capacity_quartile']
        
        # Find matching peer group
        mask = pd.Series([True] * len(self.peer_stats))
        for col in peer_cols:
            val = str(row.get(col, 'unknown')) if pd.notna(row.get(col)) else 'unknown'
            mask = mask & (self.peer_stats[col] == val)
        
        matching = self.peer_stats[mask]
        
        if len(matching) > 0:
            return matching['peer_mean'].iloc[0]
        else:
            return self.train_df['avg_adr'].median()
    
    def _get_feature_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Gets feature-based predictions."""
        df_encoded = df.copy()
        
        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                le = self.feature_encoders.get(col)
                if le:
                    df_encoded[col] = df_encoded[col].astype(str).fillna('unknown')
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        
        # Select features
        X = df_encoded[self.feature_cols].fillna(0)
        
        # Scale
        competitor_features = ['competitor_median', 'competitor_mean', 'competitor_std', 'competitor_count']
        numeric_cols = [c for c in NUMERIC_FEATURES + competitor_features if c in X.columns]
        X[numeric_cols] = self.feature_scaler.transform(X[numeric_cols])
        
        return np.exp(self.feature_model.predict(X))
    
    def predict(self, test_df: pd.DataFrame, method: str = 'weighted') -> np.ndarray:
        """
        Predicts prices using ensemble.
        
        Args:
            test_df: Test data
            method: 'simple', 'weighted', or 'stacking'
        """
        print(f"\nGenerating ensemble predictions ({method})...")
        
        # Add competitor features from training data
        test_df = add_competitor_features_from_reference(test_df, self.train_df, radius_km=10.0)
        
        n = len(test_df)
        
        # Get predictions from each model
        print("   Getting KNN predictions...", flush=True)
        knn_preds = []
        for idx, row in test_df.iterrows():
            knn_preds.append(self._get_knn_prediction(row))
        knn_preds = np.array(knn_preds)
        
        print("   Getting peer predictions...", flush=True)
        peer_preds = []
        for idx, row in test_df.iterrows():
            peer_preds.append(self._get_peer_prediction(row))
        peer_preds = np.array(peer_preds)
        
        print("   Getting feature predictions...", flush=True)
        feature_preds = self._get_feature_prediction(test_df)
        
        # Combine predictions
        if method == 'simple':
            predictions = (knn_preds + peer_preds + feature_preds) / 3
        
        elif method == 'weighted':
            # Weights based on inverse MAPE (from evaluation)
            # KNN: 21%, Peer: 30%, Feature: 24%
            w_knn = 1/21
            w_peer = 1/30
            w_feature = 1/24
            total = w_knn + w_peer + w_feature
            
            predictions = (
                w_knn/total * knn_preds +
                w_peer/total * peer_preds +
                w_feature/total * feature_preds
            )
        
        elif method == 'stacking':
            # Simple average for now
            predictions = (knn_preds + feature_preds) / 2
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.clip(predictions, 50, 250)
    
    def save(self, path: Path = MODEL_DIR) -> None:
        """Saves the ensemble model."""
        with open(path / 'ensemble_pricer.pkl', 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Saved ensemble to {path / 'ensemble_pricer.pkl'}")
    
    @classmethod
    def load(cls, path: Path = MODEL_DIR) -> 'EnsemblePricer':
        """Loads the ensemble model."""
        with open(path / 'ensemble_pricer.pkl', 'rb') as f:
            return pickle.load(f)


def evaluate_ensemble() -> Dict:
    """
    Evaluates ensemble model with different combination methods.
    """
    print("=" * 80)
    print("ENSEMBLE EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    train_df, test_df = load_and_prepare_data()
    print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Sample test for speed
    test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)
    
    # Fit ensemble
    print("\n2. Fitting ensemble...")
    ensemble = EnsemblePricer()
    ensemble.fit(train_df)
    
    # Evaluate different methods
    results = {}
    
    for method in ['simple', 'weighted']:
        print(f"\n3. Evaluating {method} ensemble...")
        
        predictions = ensemble.predict(test_sample, method=method)
        actuals = test_sample['avg_adr'].values
        
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        print(f"   MAPE: {mape:.1f}%")
        print(f"   R²: {r2:.4f}")
        
        results[method] = {'mape': mape, 'r2': r2}
    
    # Find best method
    best_method = min(results.keys(), key=lambda m: results[m]['mape'])
    print(f"\n✓ Best method: {best_method} (MAPE = {results[best_method]['mape']:.1f}%)")
    
    # Save ensemble
    ensemble.save()
    
    return results, ensemble


if __name__ == "__main__":
    results, ensemble = evaluate_ensemble()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    print("\n| Approach | MAPE |")
    print("|----------|------|")
    print("| KNN (standalone) | 21.0% |")
    print("| Relative Pricing | 26.8% |")
    print("| Competitor Features | 24.3% |")
    for method, res in results.items():
        print(f"| Ensemble ({method}) | {res['mape']:.1f}% |")
    
    best = min(results.items(), key=lambda x: x[1]['mape'])
    print(f"\n✓ Best approach: {best[0]} ensemble with MAPE = {best[1]['mape']:.1f}%")

