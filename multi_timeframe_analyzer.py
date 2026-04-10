#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时间框架分析器
Multi-Timeframe Analyzer
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    """多时间框架分析器"""
    
    def __init__(self):
        self.timeframes = {
            '1m': '1T',    # 1分钟
            '5m': '5T',    # 5分钟
            '15m': '15T',  # 15分钟
            '30m': '30T',  # 30分钟
            '1h': '1H',    # 1小时
            '4h': '4H',    # 4小时
            '1d': '1D',    # 1天
            '1w': '1W',    # 1周
            '1M': '1M'      # 1月
        }
        
    def resample_data(self, data, timeframe):
        """重采样数据到指定时间框架"""
        if timeframe not in self.timeframes:
            raise ValueError(f"不支持的时间框架: {timeframe}")
        
        freq = self.timeframes[timeframe]
        
        # 重采样OHLCV数据
        resampled = data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def calculate_indicators(self, data):
        """计算技术指标"""
        indicators = {}
        
        # 移动平均线
        indicators['SMA_5'] = talib.SMA(data['Close'], timeperiod=5)
        indicators['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
        indicators['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        indicators['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        indicators['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
        
        # 指数移动平均线
        indicators['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
        indicators['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
        
        # RSI
        indicators['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(data['Close'])
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Histogram'] = macd_hist
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'])
        indicators['BB_Upper'] = bb_upper
        indicators['BB_Middle'] = bb_middle
        indicators['BB_Lower'] = bb_lower
        
        # 随机指标
        indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
        
        # 威廉指标
        indicators['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'])
        
        # 成交量指标
        indicators['OBV'] = talib.OBV(data['Close'], data['Volume'])
        
        return pd.DataFrame(indicators, index=data.index)
    
    def analyze_trend(self, data, indicators):
        """趋势分析"""
        trend_signals = {}
        
        # 移动平均线趋势
        sma_5 = indicators['SMA_5'].iloc[-1]
        sma_20 = indicators['SMA_20'].iloc[-1]
        sma_50 = indicators['SMA_50'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # 短期趋势
        if sma_5 > sma_20:
            trend_signals['short_term'] = 'uptrend'
        elif sma_5 < sma_20:
            trend_signals['short_term'] = 'downtrend'
        else:
            trend_signals['short_term'] = 'sideways'
        
        # 中期趋势
        if sma_20 > sma_50:
            trend_signals['medium_term'] = 'uptrend'
        elif sma_20 < sma_50:
            trend_signals['medium_term'] = 'downtrend'
        else:
            trend_signals['medium_term'] = 'sideways'
        
        # 价格与均线关系
        if current_price > sma_20:
            trend_signals['price_vs_ma'] = 'above'
        else:
            trend_signals['price_vs_ma'] = 'below'
        
        return trend_signals
    
    def analyze_momentum(self, data, indicators):
        """动量分析"""
        momentum_signals = {}
        
        # RSI分析
        rsi = indicators['RSI'].iloc[-1]
        if rsi > 70:
            momentum_signals['rsi'] = 'overbought'
        elif rsi < 30:
            momentum_signals['rsi'] = 'oversold'
        else:
            momentum_signals['rsi'] = 'neutral'
        
        # MACD分析
        macd = indicators['MACD'].iloc[-1]
        macd_signal = indicators['MACD_Signal'].iloc[-1]
        macd_hist = indicators['MACD_Histogram'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0:
            momentum_signals['macd'] = 'bullish'
        elif macd < macd_signal and macd_hist < 0:
            momentum_signals['macd'] = 'bearish'
        else:
            momentum_signals['macd'] = 'neutral'
        
        # 随机指标分析
        stoch_k = indicators['STOCH_K'].iloc[-1]
        stoch_d = indicators['STOCH_D'].iloc[-1]
        
        if stoch_k > 80 and stoch_d > 80:
            momentum_signals['stoch'] = 'overbought'
        elif stoch_k < 20 and stoch_d < 20:
            momentum_signals['stoch'] = 'oversold'
        else:
            momentum_signals['stoch'] = 'neutral'
        
        return momentum_signals
    
    def analyze_volatility(self, data, indicators):
        """波动率分析"""
        volatility_signals = {}
        
        # 布林带分析
        bb_upper = indicators['BB_Upper'].iloc[-1]
        bb_lower = indicators['BB_Lower'].iloc[-1]
        bb_middle = indicators['BB_Middle'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        if current_price > bb_upper:
            volatility_signals['bb_position'] = 'above_upper'
        elif current_price < bb_lower:
            volatility_signals['bb_position'] = 'below_lower'
        else:
            volatility_signals['bb_position'] = 'within_bands'
        
        if bb_width > 0.1:  # 10%带宽
            volatility_signals['bb_width'] = 'high'
        else:
            volatility_signals['bb_width'] = 'low'
        
        # ATR分析
        atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14).iloc[-1]
        volatility_signals['atr'] = atr
        
        return volatility_signals
    
    def confluence_analysis(self, trend_signals, momentum_signals, volatility_signals):
        """汇合分析"""
        confluence_score = 0
        signals = []
        
        # 趋势信号权重
        if trend_signals['short_term'] == 'uptrend':
            confluence_score += 2
            signals.append('短期上升趋势')
        elif trend_signals['short_term'] == 'downtrend':
            confluence_score -= 2
            signals.append('短期下降趋势')
        
        if trend_signals['medium_term'] == 'uptrend':
            confluence_score += 3
            signals.append('中期上升趋势')
        elif trend_signals['medium_term'] == 'downtrend':
            confluence_score -= 3
            signals.append('中期下降趋势')
        
        # 动量信号权重
        if momentum_signals['rsi'] == 'oversold':
            confluence_score += 1
            signals.append('RSI超卖')
        elif momentum_signals['rsi'] == 'overbought':
            confluence_score -= 1
            signals.append('RSI超买')
        
        if momentum_signals['macd'] == 'bullish':
            confluence_score += 2
            signals.append('MACD看涨')
        elif momentum_signals['macd'] == 'bearish':
            confluence_score -= 2
            signals.append('MACD看跌')
        
        # 波动率信号权重
        if volatility_signals['bb_position'] == 'below_lower':
            confluence_score += 1
            signals.append('价格触及布林带下轨')
        elif volatility_signals['bb_position'] == 'above_upper':
            confluence_score -= 1
            signals.append('价格触及布林带上轨')
        
        # 生成交易信号
        if confluence_score >= 5:
            signal = 'strong_buy'
            strength = '强买入'
        elif confluence_score >= 3:
            signal = 'buy'
            strength = '买入'
        elif confluence_score <= -5:
            signal = 'strong_sell'
            strength = '强卖出'
        elif confluence_score <= -3:
            signal = 'sell'
            strength = '卖出'
        else:
            signal = 'hold'
            strength = '持有'
        
        return {
            'signal': signal,
            'strength': strength,
            'score': confluence_score,
            'signals': signals
        }
    
    def multi_timeframe_analysis(self, data, primary_timeframe='1d'):
        """多时间框架分析"""
        results = {}
        
        # 分析主要时间框架
        primary_data = self.resample_data(data, primary_timeframe)
        primary_indicators = self.calculate_indicators(primary_data)
        
        # 分析趋势
        trend_signals = self.analyze_trend(primary_data, primary_indicators)
        momentum_signals = self.analyze_momentum(primary_data, primary_indicators)
        volatility_signals = self.analyze_volatility(primary_data, primary_indicators)
        
        # 汇合分析
        confluence = self.confluence_analysis(trend_signals, momentum_signals, volatility_signals)
        
        results[primary_timeframe] = {
            'data': primary_data,
            'indicators': primary_indicators,
            'trend': trend_signals,
            'momentum': momentum_signals,
            'volatility': volatility_signals,
            'confluence': confluence
        }
        
        # 分析其他时间框架
        for tf in ['1h', '4h', '1w']:
            if tf != primary_timeframe:
                try:
                    tf_data = self.resample_data(data, tf)
                    tf_indicators = self.calculate_indicators(tf_data)
                    
                    tf_trend = self.analyze_trend(tf_data, tf_indicators)
                    tf_momentum = self.analyze_momentum(tf_data, tf_indicators)
                    tf_volatility = self.analyze_volatility(tf_data, tf_indicators)
                    
                    results[tf] = {
                        'data': tf_data,
                        'indicators': tf_indicators,
                        'trend': tf_trend,
                        'momentum': tf_momentum,
                        'volatility': tf_volatility
                    }
                except Exception as e:
                    print(f"时间框架 {tf} 分析失败: {e}")
        
        return results

def test_multi_timeframe_analyzer():
    """测试多时间框架分析器"""
    print("测试多时间框架分析器...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='1H')
    n_points = len(dates)
    
    # 生成模拟OHLCV数据
    base_price = 100
    returns = np.random.randn(n_points) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(n_points) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(n_points)) * 0.005),
        'Low': prices * (1 - np.abs(np.random.randn(n_points)) * 0.005),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_points)
    }, index=dates)
    
    # 确保High >= max(Open, Close) 和 Low <= min(Open, Close)
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    # 初始化分析器
    analyzer = MultiTimeframeAnalyzer()
    
    # 测试重采样
    print("1. 测试数据重采样...")
    daily_data = analyzer.resample_data(data, '1d')
    print(f"   原始数据点数: {len(data)}")
    print(f"   日线数据点数: {len(daily_data)}")
    
    # 测试指标计算
    print("2. 测试技术指标计算...")
    indicators = analyzer.calculate_indicators(daily_data)
    print(f"   计算了 {len(indicators.columns)} 个技术指标")
    print(f"   指标数据点数: {len(indicators)}")
    
    # 测试趋势分析
    print("3. 测试趋势分析...")
    trend_signals = analyzer.analyze_trend(daily_data, indicators)
    print(f"   短期趋势: {trend_signals['short_term']}")
    print(f"   中期趋势: {trend_signals['medium_term']}")
    print(f"   价格与均线关系: {trend_signals['price_vs_ma']}")
    
    # 测试动量分析
    print("4. 测试动量分析...")
    momentum_signals = analyzer.analyze_momentum(daily_data, indicators)
    print(f"   RSI信号: {momentum_signals['rsi']}")
    print(f"   MACD信号: {momentum_signals['macd']}")
    print(f"   随机指标信号: {momentum_signals['stoch']}")
    
    # 测试波动率分析
    print("5. 测试波动率分析...")
    volatility_signals = analyzer.analyze_volatility(daily_data, indicators)
    print(f"   布林带位置: {volatility_signals['bb_position']}")
    print(f"   布林带宽度: {volatility_signals['bb_width']}")
    print(f"   ATR值: {volatility_signals['atr']:.4f}")
    
    # 测试汇合分析
    print("6. 测试汇合分析...")
    confluence = analyzer.confluence_analysis(trend_signals, momentum_signals, volatility_signals)
    print(f"   交易信号: {confluence['signal']}")
    print(f"   信号强度: {confluence['strength']}")
    print(f"   汇合分数: {confluence['score']}")
    print(f"   信号详情: {', '.join(confluence['signals'])}")
    
    # 测试多时间框架分析
    print("7. 测试多时间框架分析...")
    results = analyzer.multi_timeframe_analysis(data, '1d')
    print(f"   分析了 {len(results)} 个时间框架")
    for tf, result in results.items():
        if 'confluence' in result:
            print(f"   {tf}: {result['confluence']['signal']} ({result['confluence']['strength']})")
        else:
            print(f"   {tf}: 数据已分析")
    
    print("多时间框架分析器测试完成！")
    return True

if __name__ == "__main__":
    test_multi_timeframe_analyzer()
