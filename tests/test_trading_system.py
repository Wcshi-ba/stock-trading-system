#!/usr/bin/env python3
"""
交易系统单元测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import StockDatabase, UserManager, TradingRecordManager, PredictionManager
from risk_management import RiskManager, StopLossManager, PositionSizer
from trading_strategies import (MomentumStrategy, MeanReversionStrategy, 
                               RSIStrategy, TradingSystem)
from stock_prediction_lstm import format_feature

class TestDatabase(unittest.TestCase):
    """数据库功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.db = StockDatabase(self.test_db_path)
        self.user_manager = UserManager(self.db)
        self.trading_manager = TradingRecordManager(self.db)
        self.prediction_manager = PredictionManager(self.db)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_user_registration(self):
        """测试用户注册"""
        result = self.user_manager.register_user("test_user", "password123", "test@example.com")
        self.assertTrue(result['success'])
        self.assertIn('user_id', result)
        
        # 测试重复注册
        result2 = self.user_manager.register_user("test_user", "password456")
        self.assertFalse(result2['success'])
        self.assertIn('已存在', result2['message'])
    
    def test_user_login(self):
        """测试用户登录"""
        # 先注册用户
        self.user_manager.register_user("test_user", "password123")
        
        # 正确登录
        result = self.user_manager.login_user("test_user", "password123")
        self.assertTrue(result['success'])
        self.assertEqual(result['user']['username'], "test_user")
        
        # 错误密码
        result2 = self.user_manager.login_user("test_user", "wrong_password")
        self.assertFalse(result2['success'])
    
    def test_trading_record(self):
        """测试交易记录"""
        # 创建用户
        user_result = self.user_manager.register_user("trader", "password123")
        user_id = user_result['user_id']
        
        # 记录交易
        success = self.trading_manager.record_trade(
            user_id=user_id,
            ticker="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            strategy="momentum",
            confidence=0.85
        )
        self.assertTrue(success)
        
        # 获取交易记录
        trades = self.trading_manager.get_user_trades(user_id)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]['ticker'], "AAPL")
        self.assertEqual(trades.iloc[0]['action'], "BUY")
    
    def test_prediction_storage(self):
        """测试预测结果存储"""
        success = self.prediction_manager.store_prediction(
            ticker="AAPL",
            predicted_price=155.0,
            prediction_date="2024-01-01",
            model_type="LSTM",
            confidence=0.75,
            features={"ma5": 150.0, "rsi": 65.0}
        )
        self.assertTrue(success)

class TestRiskManagement(unittest.TestCase):
    """风险管理测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        # 生成测试收益率数据
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        self.prices = pd.Series([100 * (1 + self.returns[:i+1]).prod() for i in range(252)])
        self.risk_manager = RiskManager()
    
    def test_var_calculation(self):
        """测试VaR计算"""
        var_hist = self.risk_manager.calculate_var(self.returns, method='historical')
        var_param = self.risk_manager.calculate_var(self.returns, method='parametric')
        var_mc = self.risk_manager.calculate_var(self.returns, method='monte_carlo')
        
        # VaR应该是负数（表示损失）
        self.assertLess(var_hist, 0)
        self.assertLess(var_param, 0)
        self.assertLess(var_mc, 0)
        
        # 历史法和参数法结果应该相近
        self.assertAlmostEqual(var_hist, var_param, delta=0.01)
    
    def test_expected_shortfall(self):
        """测试期望损失计算"""
        es = self.risk_manager.calculate_expected_shortfall(self.returns)
        var = self.risk_manager.calculate_var(self.returns)
        
        # ES应该小于（更负）VaR
        self.assertLess(es, var)
    
    def test_maximum_drawdown(self):
        """测试最大回撤计算"""
        dd_info = self.risk_manager.calculate_maximum_drawdown(self.prices)
        
        self.assertIn('max_drawdown', dd_info)
        self.assertIn('max_drawdown_date', dd_info)
        self.assertLessEqual(dd_info['max_drawdown'], 0)  # 回撤应该是负数或零
    
    def test_sharpe_ratio(self):
        """测试夏普比率计算"""
        sharpe = self.risk_manager.calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)
        # 对于随机数据，夏普比率应该接近0
        self.assertAlmostEqual(sharpe, 0, delta=1.0)
    
    def test_stop_loss_manager(self):
        """测试止损管理"""
        stop_manager = StopLossManager()
        
        # 设置止损
        order = stop_manager.set_stop_loss("AAPL", 100.0, 0.05)
        self.assertEqual(order['entry_price'], 100.0)
        self.assertEqual(order['stop_price'], 95.0)
        
        # 测试价格上涨（不触发止损）
        updated = stop_manager.update_stop_loss("AAPL", 105.0)
        self.assertTrue(updated['active'])
        
        # 测试价格下跌触发止损
        triggered = stop_manager.update_stop_loss("AAPL", 94.0)
        self.assertFalse(triggered['active'])
        self.assertTrue(triggered.get('triggered', False))
    
    def test_position_sizer(self):
        """测试仓位管理"""
        sizer = PositionSizer(100000)
        
        # 凯利公式测试
        kelly_size = sizer.kelly_criterion(0.6, 0.05, 0.03)
        self.assertGreaterEqual(kelly_size, 0)
        self.assertLessEqual(kelly_size, 0.25)  # 限制最大25%
        
        # 固定比例测试
        fixed_size = sizer.fixed_fractional(0.02)
        self.assertEqual(fixed_size, 2000)  # 2% of 100000
        
        # 波动率调整测试
        vol_size = sizer.volatility_based_sizing(0.20, 0.15)
        self.assertEqual(vol_size, 0.75)  # 15% / 20%

class TestTradingStrategies(unittest.TestCase):
    """交易策略测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        # 生成测试数据
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # 添加技术指标
        self.test_data['MA5'] = self.test_data['Close'].rolling(window=5).mean()
        self.test_data['MA20'] = self.test_data['Close'].rolling(window=20).mean()
        
        # 计算RSI
        delta = self.test_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.test_data['RSI'] = 100 - (100 / (1 + rs))
    
    def test_momentum_strategy(self):
        """测试动量策略"""
        strategy = MomentumStrategy(lookback_period=10, threshold=0.02)
        signals = strategy.generate_signals(self.test_data)
        
        # 信号应该是-1, 0, 1
        unique_signals = set(signals.unique())
        self.assertTrue(unique_signals.issubset({-1, 0, 1}))
        
        # 运行回测
        result = strategy.backtest(self.test_data)
        self.assertIn('total_return', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('max_drawdown', result)
    
    def test_mean_reversion_strategy(self):
        """测试均值回归策略"""
        strategy = MeanReversionStrategy(window=20, num_std=2.0)
        signals = strategy.generate_signals(self.test_data)
        
        # 检查信号格式
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.test_data))
        
        # 运行回测
        result = strategy.backtest(self.test_data)
        self.assertIsInstance(result['total_return'], float)
    
    def test_rsi_strategy(self):
        """测试RSI策略"""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(self.test_data)
        
        # 检查信号
        self.assertIsInstance(signals, pd.Series)
        
        # 运行回测
        result = strategy.backtest(self.test_data)
        self.assertIn('strategy_name', result)
        self.assertEqual(result['strategy_name'], "RSI Strategy")
    
    def test_trading_system(self):
        """测试交易系统"""
        system = TradingSystem()
        
        # 添加策略
        system.add_strategy(MomentumStrategy())
        system.add_strategy(MeanReversionStrategy())
        
        # 运行回测
        results = system.run_backtest(self.test_data)
        
        self.assertEqual(len(results), 2)
        self.assertIn('Momentum Strategy', results)
        self.assertIn('Mean Reversion Strategy', results)
        
        # 比较策略
        comparison = system.compare_strategies()
        self.assertEqual(len(comparison), 2)
        
        # 获取最佳策略
        best_name, best_result = system.get_best_strategy()
        self.assertIsNotNone(best_name)
        self.assertIsNotNone(best_result)

class TestDataProcessing(unittest.TestCase):
    """数据处理测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, len(dates)),
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # 确保High >= Close >= Low
        for i in range(len(self.test_data)):
            high = self.test_data.loc[i, 'High']
            low = self.test_data.loc[i, 'Low']
            close = self.test_data.loc[i, 'Close']
            
            self.test_data.loc[i, 'High'] = max(high, close)
            self.test_data.loc[i, 'Low'] = min(low, close)
    
    def test_format_feature(self):
        """测试特征工程"""
        # 添加必要的技术指标列
        self.test_data['MA5'] = self.test_data['Close'].rolling(window=5).mean()
        self.test_data['MA10'] = self.test_data['Close'].rolling(window=10).mean()
        self.test_data['MA20'] = self.test_data['Close'].rolling(window=20).mean()
        
        # 添加其他必需的列
        required_columns = ['Year', 'Month', 'Day', 'RSI', 'MACD', 'VWAP', 'SMA', 
                           'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 
                           'ATR', 'Close_yes', 'Open_yes', 'High_yes', 'Low_yes']
        
        for col in required_columns:
            if col not in self.test_data.columns:
                if col in ['Year', 'Month', 'Day']:
                    self.test_data[col] = getattr(pd.to_datetime(self.test_data['Date']), col.lower())
                else:
                    self.test_data[col] = np.random.uniform(0, 100, len(self.test_data))
        
        # 设置Date为索引
        self.test_data.set_index('Date', inplace=True)
        
        try:
            X, y = format_feature(self.test_data)
            
            # 检查输出格式
            self.assertIsInstance(X, pd.DataFrame)
            self.assertIsInstance(y, pd.Series)
            
            # 检查数据长度
            self.assertEqual(len(X), len(y))
            self.assertGreater(len(X), 0)
            
        except Exception as e:
            self.fail(f"format_feature函数执行失败: {e}")

class TestPerformanceMetrics(unittest.TestCase):
    """性能指标测试"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        self.predictions = np.random.uniform(0.8, 1.2, 100)
        self.actuals = np.random.uniform(0.8, 1.2, 100)
    
    def test_accuracy_calculation(self):
        """测试准确率计算"""
        # 方向准确率
        pred_direction = np.sign(np.diff(self.predictions))
        actual_direction = np.sign(np.diff(self.actuals))
        
        accuracy = np.mean(pred_direction == actual_direction)
        
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_rmse_calculation(self):
        """测试RMSE计算"""
        rmse = np.sqrt(np.mean((self.predictions - self.actuals) ** 2))
        
        self.assertGreaterEqual(rmse, 0)
        self.assertIsInstance(rmse, float)
    
    def test_mae_calculation(self):
        """测试MAE计算"""
        mae = np.mean(np.abs(self.predictions - self.actuals))
        
        self.assertGreaterEqual(mae, 0)
        self.assertIsInstance(mae, float)

def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDatabase,
        TestRiskManagement,
        TestTradingStrategies,
        TestDataProcessing,
        TestPerformanceMetrics
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()

if __name__ == "__main__":
    # 创建tests目录（如果不存在）
    os.makedirs('tests', exist_ok=True)
    
    print("🧪 开始运行股票交易系统单元测试...")
    print("=" * 60)
    
    success = run_all_tests()
    
    print("=" * 60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查代码。")
    
    print("\n测试覆盖的功能模块:")
    print("- 数据库操作（用户管理、交易记录、预测存储）")
    print("- 风险管理（VaR、止损、仓位管理）")
    print("- 交易策略（动量、均值回归、RSI等）")
    print("- 数据处理（特征工程）")
    print("- 性能指标（准确率、RMSE、MAE）")
