
"""
Feature Engineering Module
Provides data transformation capabilities: scaling, encoding, derived features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from config import config
from logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for automated transformations.
    Supports scaling, encoding, and derived metric generation.
    """
    
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.report: Dict[str, Any] = {
            "scaled_columns": [],
            "encoded_columns": [],
            "derived_features": [],
            "dropped_columns": []
        }
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        scale_numeric: bool = True,
        encode_categorical: bool = True,
        create_derived: bool = True,
        scaler_type: Optional[str] = None,
        encoder_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame
            scale_numeric: Whether to scale numeric columns
            encode_categorical: Whether to encode categorical columns
            create_derived: Whether to create derived features
            scaler_type: Type of scaler (standard, minmax, robust)
            encoder_type: Type of encoder (label, onehot)
        
        Returns:
            Transformed DataFrame and transformation report
        """
        df_transformed = df.copy()
        
        # Apply transformations
        if scale_numeric:
            df_transformed = self._scale_numeric(df_transformed, scaler_type)
        
        if encode_categorical:
            df_transformed = self._encode_categorical(df_transformed, encoder_type)
        
        if create_derived:
            df_transformed = self._create_derived_features(df_transformed)
        
        logger.info(f"Feature engineering complete: {len(df_transformed.columns)} columns")
        return df_transformed, self.report
    
    def _scale_numeric(
        self, 
        df: pd.DataFrame, 
        scaler_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Scale numeric columns using specified scaler"""
        scaler_type = scaler_type or config.analysis.DEFAULT_SCALER
        
        # Get numeric columns (exclude IDs and already scaled)
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if 'id' not in col.lower() and not col.endswith('_scaled')
        ]
        
        if not numeric_cols:
            return df
        
        # Select scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using standard")
            scaler = StandardScaler()
        
        # Apply scaling
        try:
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Create new columns with _scaled suffix
            for i, col in enumerate(numeric_cols):
                new_col = f"{col}_scaled"
                df[new_col] = scaled_data[:, i]
                self.scalers[col] = scaler
                self.report["scaled_columns"].append({
                    "original": col,
                    "scaled": new_col,
                    "method": scaler_type
                })
            
            logger.info(f"Scaled {len(numeric_cols)} numeric columns using {scaler_type}")
        
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
        
        return df
    
    def _encode_categorical(
        self, 
        df: pd.DataFrame, 
        encoder_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Encode categorical columns"""
        encoder_type = encoder_type or config.analysis.DEFAULT_ENCODER
        
        # Get categorical columns
        categorical_cols = [
            col for col in df.select_dtypes(include=['object']).columns
            if 'id' not in col.lower() and not col.endswith('_encoded')
        ]
        
        if not categorical_cols:
            return df
        
        for col in categorical_cols:
            try:
                if encoder_type == "label":
                    df = self._label_encode(df, col)
                elif encoder_type == "onehot":
                    df = self._onehot_encode(df, col)
                else:
                    logger.warning(f"Unknown encoder type: {encoder_type}, using label")
                    df = self._label_encode(df, col)
            
            except Exception as e:
                logger.error(f"Encoding failed for {col}: {e}")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns using {encoder_type}")
        return df
    
    def _label_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply label encoding to a categorical column"""
        encoder = LabelEncoder()
        new_col = f"{col}_encoded"
        
        # Handle missing values
        df[col] = df[col].fillna('MISSING')
        
        # Encode
        df[new_col] = encoder.fit_transform(df[col].astype(str))
        self.encoders[col] = encoder
        
        self.report["encoded_columns"].append({
            "original": col,
            "encoded": new_col,
            "method": "label",
            "n_classes": len(encoder.classes_)
        })
        
        return df
    
    def _onehot_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply one-hot encoding to a categorical column"""
        # Limit to reasonable cardinality
        n_unique = df[col].nunique()
        if n_unique > 20:
            logger.warning(f"Column {col} has {n_unique} unique values, using label encoding instead")
            return self._label_encode(df, col)
        
        # One-hot encode
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        df = pd.concat([df, dummies], axis=1)
        
        self.report["encoded_columns"].append({
            "original": col,
            "method": "onehot",
            "n_columns_created": len(dummies.columns)
        })
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df
        
        # Limit to avoid explosion
        numeric_cols = numeric_cols[:10]
        
        # Sum of all numeric columns
        try:
            df['num_sum'] = df[numeric_cols].sum(axis=1)
            self.report["derived_features"].append({
                "name": "num_sum",
                "description": "Sum of numeric columns"
            })
        except Exception:
            pass
        
        # Mean of all numeric columns
        try:
            df['num_mean'] = df[numeric_cols].mean(axis=1)
            self.report["derived_features"].append({
                "name": "num_mean",
                "description": "Mean of numeric columns"
            })
        except Exception:
            pass
        
        # Standard deviation
        try:
            df['num_std'] = df[numeric_cols].std(axis=1)
            self.report["derived_features"].append({
                "name": "num_std",
                "description": "Std deviation of numeric columns"
            })
        except Exception:
            pass
        
        # Min and max
        try:
            df['num_min'] = df[numeric_cols].min(axis=1)
            df['num_max'] = df[numeric_cols].max(axis=1)
            df['num_range'] = df['num_max'] - df['num_min']
            self.report["derived_features"].extend([
                {"name": "num_min", "description": "Minimum value"},
                {"name": "num_max", "description": "Maximum value"},
                {"name": "num_range", "description": "Range (max - min)"}
            ])
        except Exception:
            pass
        
        logger.info(f"Created {len(self.report['derived_features'])} derived features")
        return df
    
    def get_feature_importance_proxy(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Calculate simple correlation-based feature importance.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
        
        Returns:
            DataFrame with feature importance scores
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if not numeric_cols:
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = []
        for col in numeric_cols:
            try:
                corr = df[col].corr(df[target_col])
                if pd.notna(corr):
                    correlations.append({
                        "feature": col,
                        "correlation": abs(corr),
                        "direction": "positive" if corr > 0 else "negative"
                    })
            except Exception:
                pass
        
        importance_df = pd.DataFrame(correlations)
        if not importance_df.empty:
            importance_df = importance_df.sort_values("correlation", ascending=False)
        
        return importance_df
