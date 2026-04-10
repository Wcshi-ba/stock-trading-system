#!/usr/bin/env python3
"""
A股特征工程模块
为A股数据添加技术指标和特征，使其与美股数据格式兼容
"""

import pandas as pd
import numpy as np
from typing import Tuple
import talib

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    为A股数据添加技术指标
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        pd.DataFrame: 添加技术指标后的DataFrame
    """
    df = df.copy()
    
    # 确保数据按日期排序
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 提取OHLCV数据
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # 1. 移动平均线
    df['MA5'] = talib.SMA(close, timeperiod=5)
    df['MA10'] = talib.SMA(close, timeperiod=10)
    df['MA20'] = talib.SMA(close, timeperiod=20)
    df['MA50'] = talib.SMA(close, timeperiod=50)
    
    # 2. RSI相对强弱指数
    df['RSI'] = talib.RSI(close, timeperiod=14)
    
    # 3. MACD指标
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal_Line'] = macd_signal
    df['MACD_Histogram'] = macd_hist
    
    # 4. 布林带
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['Upper_band'] = bb_upper
    df['SMA'] = bb_middle  # 中轨
    df['Lower_band'] = bb_lower
    
    # 5. 标准差
    df['Std_dev'] = talib.STDDEV(close, timeperiod=20)
    
    # 6. ATR平均真实波幅
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    # 7. VWAP成交量加权平均价
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    # 8. 相对表现
    df['Relative_Performance'] = df['Close'].pct_change()
    
    # 9. ROC变化率
    df['ROC'] = talib.ROC(close, timeperiod=10)
    
    # 10. 添加日期特征
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # 11. 前一日数据
    df['Close_yes'] = df['Close'].shift(1)
    df['Open_yes'] = df['Open'].shift(1)
    df['High_yes'] = df['High'].shift(1)
    df['Low_yes'] = df['Low'].shift(1)
    
    # 12. 成交量相关指标
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # 13. 价格相关指标
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
    df['Gap'] = df['Open'] - df['Close_yes']
    df['Gap_Pct'] = df['Gap'] / df['Close_yes']
    
    # 14. 波动率指标
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volatility_Pct'] = df['Volatility'] / df['Close']
    
    # 15. 动量指标
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Momentum_Pct'] = df['Momentum'] / df['Close'].shift(10)
    
    # 16. 趋势指标
    df['Trend_5'] = np.where(df['Close'] > df['MA5'], 1, 0)
    df['Trend_10'] = np.where(df['Close'] > df['MA10'], 1, 0)
    df['Trend_20'] = np.where(df['Close'] > df['MA20'], 1, 0)
    
    # 17. 支撑阻力位
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
    df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
    
    # 18. 成交量价格趋势
    df['VPT'] = (df['Volume'] * df['Relative_Performance']).cumsum()
    
    # 19. 资金流向指标
    df['Money_Flow'] = df['Close'] * df['Volume']
    df['Money_Flow_MA'] = df['Money_Flow'].rolling(window=20).mean()
    
    # 20. 市场情绪指标
    df['Fear_Greed'] = (df['RSI'] - 50) / 50  # -1到1之间，-1表示极度恐惧，1表示极度贪婪
    
    return df

def process_a_share_data(csv_path: str, output_path: str = None) -> str:
    """
    处理A股数据，添加技术指标
    
    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出CSV文件路径，如果为None则覆盖原文件
        
    Returns:
        str: 处理后的文件路径
    """
    try:
        # 读取A股数据
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"原始数据: {len(df)} 行")
        
        # 添加技术指标
        df_enhanced = add_technical_indicators(df)
        
        # 删除包含NaN的行（技术指标计算需要的历史数据）
        df_enhanced = df_enhanced.dropna()
        
        print(f"特征工程后: {len(df_enhanced)} 行")
        print(f"特征数量: {len(df_enhanced.columns)} 个")
        
        # 保存处理后的数据
        if output_path is None:
            output_path = csv_path
            
        df_enhanced.to_csv(output_path, index=False)
        
        print(f"[OK] A股特征工程完成: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] A股特征工程失败: {e}")
        raise

def validate_a_share_features(df: pd.DataFrame) -> bool:
    """
    验证A股数据是否包含必要的特征
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: 是否包含所有必要特征
    """
    required_features = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal_Line',
        'Upper_band', 'SMA', 'Lower_band', 'Std_dev', 'ATR',
        'VWAP', 'Relative_Performance', 'ROC', 'Year', 'Month', 'Day',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    
    missing_features = [col for col in required_features if col not in df.columns]
    
    if missing_features:
        print(f"缺少必要特征: {missing_features}")
        return False
    
    return True

if __name__ == "__main__":
    # 测试A股特征工程
    print("=== A股特征工程测试 ===")
    
    # 处理贵州茅台数据
    input_file = "data/600519.csv"
    output_file = "data/600519_enhanced.csv"
    
    try:
        result_path = process_a_share_data(input_file, output_file)
        
        # 验证结果
        df_result = pd.read_csv(result_path)
        print(f"\n处理结果验证:")
        print(f"数据行数: {len(df_result)}")
        print(f"特征数量: {len(df_result.columns)}")
        
        if validate_a_share_features(df_result):
            print("[OK] 特征验证通过")
        else:
            print("[ERROR] 特征验证失败")
            
        # 显示前几行数据
        print(f"\n前5行数据:")
        print(df_result.head())
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
