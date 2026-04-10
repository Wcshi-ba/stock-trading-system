#!/usr/bin/env python3
"""
多策略交易系统
包含动量策略、均值回归策略、技术指标策略等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("警告: TA-Lib未安装，将使用简化的技术指标计算")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BaseStrategy(ABC):
    """交易策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.positions = []
        self.returns = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        pass
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """策略回测"""
        signals = self.generate_signals(data)
        
        # 计算收益
        returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        # 计算累积收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算性能指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'strategy_name': self.name,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'signals': signals,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns
        }

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基于价格动量生成信号
        当价格相对于N日前上涨超过阈值时买入，下跌超过阈值时卖出
        """
        price_change = data['Close'].pct_change(self.lookback_period)
        
        signals = pd.Series(0, index=data.index)
        signals[price_change > self.threshold] = 1  # 买入信号
        signals[price_change < -self.threshold] = -1  # 卖出信号
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__("Mean Reversion Strategy")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基于布林带的均值回归策略
        当价格触及下轨时买入，触及上轨时卖出
        """
        # 计算布林带
        rolling_mean = data['Close'].rolling(window=self.window).mean()
        rolling_std = data['Close'].rolling(window=self.window).std()
        
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] <= lower_band] = 1  # 买入信号
        signals[data['Close'] >= upper_band] = -1  # 卖出信号
        
        return signals

class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Strategy")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基于RSI指标的交易策略
        RSI < 30时买入，RSI > 70时卖出
        """
        if 'RSI' in data.columns:
            rsi = data['RSI']
        else:
            if HAS_TALIB:
                rsi = talib.RSI(data['Close'].values, timeperiod=self.period)
                rsi = pd.Series(rsi, index=data.index)
            else:
                # 简化的RSI计算
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # 买入信号
        signals[rsi > self.overbought] = -1  # 卖出信号
        
        return signals

class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基于MACD指标的交易策略
        MACD线上穿信号线时买入，下穿时卖出
        """
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            macd = data['MACD']
            signal_line = data['Signal_Line']
        else:
            if HAS_TALIB:
                macd, signal_line, _ = talib.MACD(data['Close'].values, 
                                            fastperiod=self.fast_period,
                                            slowperiod=self.slow_period, 
                                            signalperiod=self.signal_period)
                macd = pd.Series(macd, index=data.index)
                signal_line = pd.Series(signal_line, index=data.index)
            else:
                # 简化的MACD计算
                exp1 = data['Close'].ewm(span=self.fast_period).mean()
                exp2 = data['Close'].ewm(span=self.slow_period).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=self.signal_period).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # MACD上穿信号线
        macd_cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        # MACD下穿信号线
        macd_cross_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        signals[macd_cross_up] = 1  # 买入信号
        signals[macd_cross_down] = -1  # 卖出信号
        
        return signals

class MovingAverageCrossoverStrategy(BaseStrategy):
    """移动平均线交叉策略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__("MA Crossover Strategy")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        双移动平均线交叉策略
        短期均线上穿长期均线时买入，下穿时卖出
        """
        if f'MA{self.short_window}' in data.columns:
            short_ma = data[f'MA{self.short_window}']
        else:
            short_ma = data['Close'].rolling(window=self.short_window).mean()
        
        if f'MA{self.long_window}' in data.columns:
            long_ma = data[f'MA{self.long_window}']
        else:
            long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # 金叉：短期均线上穿长期均线
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        # 死叉：短期均线下穿长期均线
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        signals[golden_cross] = 1  # 买入信号
        signals[death_cross] = -1  # 卖出信号
        
        return signals

class BreakoutStrategy(BaseStrategy):
    """突破策略"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Breakout Strategy")
        self.lookback_period = lookback_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        价格突破策略
        突破N日最高价时买入，跌破N日最低价时卖出
        """
        rolling_max = data['High'].rolling(window=self.lookback_period).max()
        rolling_min = data['Low'].rolling(window=self.lookback_period).min()
        
        signals = pd.Series(0, index=data.index)
        
        # 突破最高价
        breakout_up = data['Close'] > rolling_max.shift(1)
        # 跌破最低价
        breakout_down = data['Close'] < rolling_min.shift(1)
        
        signals[breakout_up] = 1  # 买入信号
        signals[breakout_down] = -1  # 卖出信号
        
        return signals

class MLStrategy(BaseStrategy):
    """机器学习策略"""
    
    def __init__(self, lookback_period: int = 30, prediction_horizon: int = 1):
        super().__init__("ML Strategy")
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备机器学习特征"""
        features = pd.DataFrame(index=data.index)
        
        # 价格特征
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # 技术指标特征
        if 'RSI' in data.columns:
            features['rsi'] = data['RSI']
        if 'MACD' in data.columns:
            features['macd'] = data['MACD']
        if 'MA5' in data.columns:
            features['ma5_ratio'] = data['Close'] / data['MA5']
        if 'MA20' in data.columns:
            features['ma20_ratio'] = data['Close'] / data['MA20']
        
        # 波动率特征
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # 动量特征
        features['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        features['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        
        # 成交量特征
        if 'Volume' in data.columns:
            features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        return features.dropna()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基于机器学习的交易策略
        """
        features = self.prepare_features(data)
        
        # 创建标签：未来N日收益率 > 0为1，否则为0
        future_returns = data['Close'].shift(-self.prediction_horizon) / data['Close'] - 1
        labels = (future_returns > 0).astype(int)
        
        # 对齐特征和标签
        aligned_data = pd.concat([features, labels], axis=1, join='inner')
        aligned_data.columns = list(features.columns) + ['label']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < self.lookback_period * 2:
            return pd.Series(0, index=data.index)
        
        signals = pd.Series(0, index=data.index)
        
        # 滚动训练和预测
        for i in range(self.lookback_period, len(aligned_data) - self.prediction_horizon):
            # 训练数据
            train_start = max(0, i - self.lookback_period * 2)
            train_end = i
            
            X_train = aligned_data.iloc[train_start:train_end, :-1]
            y_train = aligned_data.iloc[train_start:train_end, -1]
            
            # 预测数据
            X_pred = aligned_data.iloc[i:i+1, :-1]
            
            if len(X_train) < 20:  # 确保有足够的训练数据
                continue
            
            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # 训练模型
            self.model.fit(X_train_scaled, y_train)
            
            # 预测
            pred_proba = self.model.predict_proba(X_pred_scaled)[0]
            
            # 生成信号
            if len(pred_proba) > 1:
                confidence = pred_proba[1]  # 上涨概率
                if confidence > 0.6:
                    signals.iloc[i] = 1  # 买入
                elif confidence < 0.4:
                    signals.iloc[i] = -1  # 卖出
        
        return signals

class EnsembleStrategy(BaseStrategy):
    """集成策略"""
    
    def __init__(self, strategies: List[BaseStrategy], weights: List[float] = None):
        super().__init__("Ensemble Strategy")
        self.strategies = strategies
        self.weights = weights or [1/len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("权重数量必须与策略数量相等")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        集成多个策略的信号
        """
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        # 加权平均
        ensemble_signals = pd.Series(0.0, index=data.index)
        for signals, weight in zip(all_signals, self.weights):
            ensemble_signals += signals * weight
        
        # 转换为离散信号
        final_signals = pd.Series(0, index=data.index)
        final_signals[ensemble_signals > 0.3] = 1
        final_signals[ensemble_signals < -0.3] = -1
        
        return final_signals

class TradingSystem:
    """交易系统管理类"""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        self.strategies[strategy.name] = strategy
    
    def run_backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """运行所有策略的回测"""
        results = {}
        
        for name, strategy in self.strategies.items():
            print(f"运行策略: {name}")
            result = strategy.backtest(data, initial_capital)
            results[name] = result
        
        self.results = results
        return results
    
    def compare_strategies(self) -> pd.DataFrame:
        """比较策略表现"""
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Strategy': name,
                'Total Return': f"{result['total_return']:.2%}",
                'Annual Return': f"{result['annual_return']:.2%}",
                'Volatility': f"{result['volatility']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}"
            })
        
        return pd.DataFrame(comparison)
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Tuple[str, Dict]:
        """获取最佳策略"""
        if not self.results:
            return None, None
        
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x][metric])
        return best_name, self.results[best_name]

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # 生成模拟股价数据
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # 添加技术指标
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = talib.RSI(data['Close'].values, timeperiod=14)
    
    # 创建交易系统
    trading_system = TradingSystem()
    
    # 添加策略
    trading_system.add_strategy(MomentumStrategy(lookback_period=10))
    trading_system.add_strategy(MeanReversionStrategy(window=20))
    trading_system.add_strategy(RSIStrategy())
    trading_system.add_strategy(MovingAverageCrossoverStrategy(5, 20))
    
    # 运行回测
    results = trading_system.run_backtest(data)
    
    # 比较策略
    comparison = trading_system.compare_strategies()
    print("\n=== 策略比较 ===")
    print(comparison)
    
    # 获取最佳策略
    best_name, best_result = trading_system.get_best_strategy()
    print(f"\n最佳策略: {best_name}")
    print(f"夏普比率: {best_result['sharpe_ratio']:.3f}")
