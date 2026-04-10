#!/usr/bin/env python3
"""
风险评估与管理模块
提供VaR计算、止损机制、投资组合风险分析等功能
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """风险管理类"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_var(self, returns: pd.Series, method: str = 'historical') -> float:
        """
        计算风险价值(Value at Risk)
        
        Args:
            returns: 收益率序列
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR值
        """
        if method == 'historical':
            return self._historical_var(returns)
        elif method == 'parametric':
            return self._parametric_var(returns)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns)
        else:
            raise ValueError("方法必须是 'historical', 'parametric', 或 'monte_carlo'")
    
    def _historical_var(self, returns: pd.Series) -> float:
        """历史模拟法计算VaR"""
        return np.percentile(returns, self.alpha * 100)
    
    def _parametric_var(self, returns: pd.Series) -> float:
        """参数法计算VaR（假设正态分布）"""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(self.alpha)
        return mean + z_score * std
    
    def _monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> float:
        """蒙特卡洛模拟法计算VaR"""
        mean = returns.mean()
        std = returns.std()
        
        # 生成随机收益率
        simulated_returns = np.random.normal(mean, std, n_simulations)
        return np.percentile(simulated_returns, self.alpha * 100)
    
    def calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """
        计算期望损失(Expected Shortfall/CVaR)
        
        Args:
            returns: 收益率序列
        
        Returns:
            ES值
        """
        var = self.calculate_var(returns)
        return returns[returns <= var].mean()
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
        
        Returns:
            包含最大回撤信息的字典
        """
        # 计算累积收益
        cumulative = (1 + prices.pct_change()).cumprod()
        
        # 计算历史最高点
        running_max = cumulative.expanding().max()
        
        # 计算回撤
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        # 找到回撤开始和结束时间
        peak_date = running_max.loc[:max_drawdown_date].idxmax()
        
        # 找到回撤恢复时间
        recovery_date = None
        peak_value = running_max.loc[peak_date]
        
        for date in cumulative.loc[max_drawdown_date:].index:
            if cumulative.loc[date] >= peak_value:
                recovery_date = date
                break
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'peak_date': peak_date,
            'recovery_date': recovery_date,
            'drawdown_duration': (max_drawdown_date - peak_date).days if peak_date else None,
            'recovery_duration': (recovery_date - max_drawdown_date).days if recovery_date else None
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（年化）
        
        Returns:
            夏普比率
        """
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        计算Beta系数
        
        Args:
            stock_returns: 股票收益率
            market_returns: 市场收益率
        
        Returns:
            Beta值
        """
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    def portfolio_risk_metrics(self, portfolio_returns: pd.Series, 
                             benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        计算投资组合风险指标
        
        Args:
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准收益率
        
        Returns:
            风险指标字典
        """
        metrics = {
            'var_95': self.calculate_var(portfolio_returns, method='historical'),
            'var_99': self.calculate_var(pd.Series(portfolio_returns), method='historical'),
            'expected_shortfall': self.calculate_expected_shortfall(portfolio_returns),
            'volatility': portfolio_returns.std() * np.sqrt(252),  # 年化波动率
            'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
        
        if benchmark_returns is not None:
            metrics['beta'] = self.calculate_beta(portfolio_returns, benchmark_returns)
            metrics['correlation'] = portfolio_returns.corr(benchmark_returns)
            
            # 计算信息比率
            active_returns = portfolio_returns - benchmark_returns
            metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        return metrics

class StopLossManager:
    """止损管理类"""
    
    def __init__(self):
        self.stop_loss_orders = {}
    
    def set_stop_loss(self, ticker: str, entry_price: float, 
                     stop_loss_pct: float = 0.05, 
                     trailing: bool = False) -> Dict[str, any]:
        """
        设置止损订单
        
        Args:
            ticker: 股票代码
            entry_price: 入场价格
            stop_loss_pct: 止损百分比
            trailing: 是否为追踪止损
        
        Returns:
            止损订单信息
        """
        stop_price = entry_price * (1 - stop_loss_pct)
        
        order = {
            'ticker': ticker,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'stop_loss_pct': stop_loss_pct,
            'trailing': trailing,
            'highest_price': entry_price,
            'active': True
        }
        
        self.stop_loss_orders[ticker] = order
        return order
    
    def update_stop_loss(self, ticker: str, current_price: float) -> Dict[str, any]:
        """
        更新止损价格（用于追踪止损）
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
        
        Returns:
            更新后的止损信息
        """
        if ticker not in self.stop_loss_orders:
            return None
        
        order = self.stop_loss_orders[ticker]
        
        if not order['active']:
            return order
        
        # 检查是否触发止损
        if current_price <= order['stop_price']:
            order['active'] = False
            order['triggered'] = True
            order['trigger_price'] = current_price
            return order
        
        # 更新追踪止损
        if order['trailing'] and current_price > order['highest_price']:
            order['highest_price'] = current_price
            new_stop_price = current_price * (1 - order['stop_loss_pct'])
            order['stop_price'] = max(order['stop_price'], new_stop_price)
        
        return order
    
    def check_stop_loss_trigger(self, ticker: str, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            ticker: 股票代码
            current_price: 当前价格
        
        Returns:
            是否触发止损
        """
        order = self.update_stop_loss(ticker, current_price)
        return order and order.get('triggered', False)

class PositionSizer:
    """仓位管理类"""
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凯利公式计算最优仓位大小
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
        
        Returns:
            建议仓位比例
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs(avg_loss)  # 赔率
        p = win_rate  # 胜率
        q = 1 - p  # 败率
        
        kelly_pct = (b * p - q) / b
        return max(0, min(kelly_pct, 0.25))  # 限制最大25%
    
    def fixed_fractional(self, risk_per_trade: float = 0.02) -> float:
        """
        固定比例仓位管理
        
        Args:
            risk_per_trade: 每笔交易风险比例
        
        Returns:
            仓位大小
        """
        return self.total_capital * risk_per_trade
    
    def volatility_based_sizing(self, volatility: float, target_volatility: float = 0.15) -> float:
        """
        基于波动率的仓位管理
        
        Args:
            volatility: 资产波动率
            target_volatility: 目标波动率
        
        Returns:
            仓位比例
        """
        if volatility == 0:
            return 0
        
        return min(target_volatility / volatility, 1.0)

class PortfolioOptimizer:
    """投资组合优化类"""
    
    def __init__(self):
        pass
    
    def calculate_portfolio_weights(self, returns: pd.DataFrame, 
                                  method: str = 'equal_weight') -> pd.Series:
        """
        计算投资组合权重
        
        Args:
            returns: 资产收益率矩阵
            method: 优化方法 ('equal_weight', 'min_variance', 'max_sharpe')
        
        Returns:
            权重向量
        """
        n_assets = len(returns.columns)
        
        if method == 'equal_weight':
            return pd.Series(1/n_assets, index=returns.columns)
        
        elif method == 'min_variance':
            return self._min_variance_weights(returns)
        
        elif method == 'max_sharpe':
            return self._max_sharpe_weights(returns)
        
        else:
            raise ValueError("方法必须是 'equal_weight', 'min_variance', 或 'max_sharpe'")
    
    def _min_variance_weights(self, returns: pd.DataFrame) -> pd.Series:
        """最小方差组合"""
        cov_matrix = returns.cov()
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones((len(returns.columns), 1))
        
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        
        return pd.Series(weights.flatten(), index=returns.columns)
    
    def _max_sharpe_weights(self, returns: pd.DataFrame, 
                           risk_free_rate: float = 0.02) -> pd.Series:
        """最大夏普比率组合（简化版本）"""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # 简化的切点组合
        excess_returns = mean_returns - risk_free_rate
        inv_cov = np.linalg.pinv(cov_matrix)
        
        weights = inv_cov @ excess_returns
        weights = weights / weights.sum()
        
        # 确保权重为正
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=returns.columns)
    
    def calculate_portfolio_performance(self, weights: pd.Series, 
                                      returns: pd.DataFrame) -> Dict[str, float]:
        """
        计算投资组合表现
        
        Args:
            weights: 权重向量
            returns: 收益率矩阵
        
        Returns:
            表现指标
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        
        return {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    
    # 风险管理测试
    risk_manager = RiskManager()
    
    print("=== 风险指标测试 ===")
    print(f"VaR (95%): {risk_manager.calculate_var(returns):.4f}")
    print(f"Expected Shortfall: {risk_manager.calculate_expected_shortfall(returns):.4f}")
    print(f"Sharpe Ratio: {risk_manager.calculate_sharpe_ratio(returns):.4f}")
    
    # 止损管理测试
    stop_manager = StopLossManager()
    order = stop_manager.set_stop_loss('AAPL', 150.0, 0.05, trailing=True)
    print(f"\n=== 止损订单 ===")
    print(f"止损订单: {order}")
    
    # 测试价格更新
    updated_order = stop_manager.update_stop_loss('AAPL', 155.0)
    print(f"价格更新后: {updated_order}")
    
    # 仓位管理测试
    position_sizer = PositionSizer(100000)
    kelly_size = position_sizer.kelly_criterion(0.6, 0.05, 0.03)
    print(f"\n=== 仓位管理 ===")
    print(f"凯利公式建议仓位: {kelly_size:.2%}")
