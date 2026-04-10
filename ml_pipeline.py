#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习流水线
Machine Learning Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """机器学习流水线"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def create_features(self, data):
        """特征工程"""
        features = pd.DataFrame(index=data.index)
        
        # 价格特征
        features['price_change'] = data['Close'].pct_change()
        features['price_change_abs'] = features['price_change'].abs()
        features['log_price'] = np.log(data['Close'])
        
        # 移动平均特征
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['Close'].rolling(window).mean()
            features[f'sma_ratio_{window}'] = data['Close'] / features[f'sma_{window}']
        
        # 波动率特征
        features['volatility_5'] = data['Close'].rolling(5).std()
        features['volatility_20'] = data['Close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # 成交量特征
        features['volume_ma_5'] = data['Volume'].rolling(5).mean()
        features['volume_ma_20'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_ma_20']
        
        # 价格位置特征
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # 技术指标特征
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 布林带
        bb_middle = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # 时间特征
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # 滞后特征
        for lag in [1, 2, 3, 5, 10]:
            features[f'price_lag_{lag}'] = data['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
        
        # 未来收益率（目标变量）
        features['future_return_1'] = data['Close'].shift(-1) / data['Close'] - 1
        features['future_return_5'] = data['Close'].shift(-5) / data['Close'] - 1
        features['future_return_10'] = data['Close'].shift(-10) / data['Close'] - 1
        
        return features.dropna()
    
    def prepare_data(self, features, target_col='future_return_1'):
        """准备训练数据"""
        # 分离特征和目标
        X = features.drop([col for col in features.columns if 'future_return' in col], axis=1)
        y = features[target_col]
        
        # 处理无穷大和NaN值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, X.columns, scaler
    
    def train_models(self, X, y, test_size=0.2):
        """训练多个模型"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"训练 {name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred
            }
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"  RMSE: {rmse:.6f}")
            print(f"  R2: {r2:.4f}")
            print(f"  CV RMSE: {cv_rmse:.6f}")
        
        return results, X_test, y_test
    
    def ensemble_prediction(self, models_results, X):
        """集成预测"""
        predictions = []
        weights = []
        
        for name, result in models_results.items():
            pred = result['model'].predict(X)
            predictions.append(pred)
            # 使用R²作为权重
            weights.append(max(0, result['r2']))
        
        # 加权平均
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def hyperparameter_tuning(self, X, y, model_name='RandomForest'):
        """超参数调优"""
        if model_name == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            return None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def feature_selection(self, X, y, feature_names, top_k=20):
        """特征选择"""
        # 使用随机森林进行特征重要性排序
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 获取特征重要性
        importance = rf.feature_importances_
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前k个特征
        selected_features = [f[0] for f in feature_importance[:top_k]]
        selected_indices = [feature_names.tolist().index(f) for f in selected_features]
        
        return selected_features, selected_indices
    
    def model_interpretation(self, model, feature_names, top_n=10):
        """模型解释"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'top_features': feature_importance[:top_n],
                'feature_importance': dict(feature_importance)
            }
        else:
            return None
    
    def full_pipeline(self, data, target_col='future_return_1'):
        """完整流水线"""
        print("开始机器学习流水线...")
        
        # 1. 特征工程
        print("1. 特征工程...")
        features = self.create_features(data)
        print(f"   生成了 {features.shape[1]} 个特征")
        
        # 2. 数据准备
        print("2. 数据准备...")
        X, y, feature_names, scaler = self.prepare_data(features, target_col)
        print(f"   训练样本数: {X.shape[0]}")
        print(f"   特征数: {X.shape[1]}")
        
        # 3. 特征选择
        print("3. 特征选择...")
        selected_features, selected_indices = self.feature_selection(X, y, feature_names)
        X_selected = X[:, selected_indices]
        print(f"   选择了 {len(selected_features)} 个重要特征")
        
        # 4. 模型训练
        print("4. 模型训练...")
        models_results, X_test, y_test = self.train_models(X_selected, y)
        
        # 5. 集成预测
        print("5. 集成预测...")
        ensemble_pred = self.ensemble_prediction(models_results, X_test)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        print(f"   集成模型 RMSE: {ensemble_rmse:.6f}")
        print(f"   集成模型 R2: {ensemble_r2:.4f}")
        
        # 6. 超参数调优（示例）
        print("6. 超参数调优...")
        best_rf, best_params = self.hyperparameter_tuning(X_selected, y, 'RandomForest')
        if best_rf is not None:
            print(f"   最佳参数: {best_params}")
        
        # 7. 模型解释
        print("7. 模型解释...")
        best_model = max(models_results.items(), key=lambda x: x[1]['r2'])
        interpretation = self.model_interpretation(best_model[1]['model'], 
                                                 np.array(selected_features))
        if interpretation:
            print("   最重要的特征:")
            for feature, importance in interpretation['top_features'][:5]:
                print(f"     {feature}: {importance:.4f}")
        
        return {
            'features': features,
            'selected_features': selected_features,
            'models_results': models_results,
            'ensemble_prediction': ensemble_pred,
            'ensemble_metrics': {'rmse': ensemble_rmse, 'r2': ensemble_r2},
            'interpretation': interpretation,
            'scaler': scaler
        }

def test_ml_pipeline():
    """测试机器学习流水线"""
    print("测试机器学习流水线...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n_points = len(dates)
    
    # 生成模拟股票数据
    base_price = 100
    returns = np.random.randn(n_points) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(n_points) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(n_points)) * 0.005),
        'Low': prices * (1 - np.abs(np.random.randn(n_points)) * 0.005),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_points)
    }, index=dates)
    
    # 确保OHLC逻辑正确
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    # 初始化流水线
    pipeline = MLPipeline()
    
    # 运行完整流水线
    results = pipeline.full_pipeline(data)
    
    print("机器学习流水线测试完成！")
    return True

if __name__ == "__main__":
    test_ml_pipeline()
