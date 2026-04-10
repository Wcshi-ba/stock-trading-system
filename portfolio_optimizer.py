#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合优化引擎
Portfolio Optimization Engine
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self):
        self.returns = None
        self.cov_matrix = None
        self.expected_returns = None
        
    def prepare_data(self, price_data):
        """准备数据"""
        # 计算收益率
        returns = price_data.pct_change().dropna()
        self.returns = returns
        self.expected_returns = returns.mean() * 252  # 年化收益率
        self.cov_matrix = returns.cov() * 252  # 年化协方差矩阵
        return returns
    
    def mean_variance_optimization(self, target_return=None, risk_free_rate=0.02):
        """均值方差优化"""
        n_assets = len(self.expected_returns)
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重和为1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(x * self.expected_returns) - target_return
            })
        
        # 边界条件
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 初始权重
        x0 = np.array([1/n_assets] * n_assets)
        
        # 目标函数：最小化方差
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # 优化
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            
            return {
                'weights': dict(zip(self.expected_returns.index, weights)),
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
        else:
            return {'success': False, 'message': '优化失败'}
    
    def risk_parity_optimization(self):
        """风险平价优化"""
        n_assets = len(self.expected_returns)
        
        def objective(weights):
            # 风险贡献
            risk_contrib = weights * np.sqrt(np.diag(self.cov_matrix))
            # 风险贡献的方差
            return np.var(risk_contrib)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            return {
                'weights': dict(zip(self.expected_returns.index, weights)),
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'success': True
            }
        else:
            return {'success': False, 'message': '风险平价优化失败'}
    
    def minimum_variance_optimization(self):
        """最小方差优化"""
        n_assets = len(self.expected_returns)
        
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            return {
                'weights': dict(zip(self.expected_returns.index, weights)),
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'success': True
            }
        else:
            return {'success': False, 'message': '最小方差优化失败'}
    
    def efficient_frontier(self, num_portfolios=100):
        """有效前沿"""
        n_assets = len(self.expected_returns)
        results = []
        
        for _ in range(num_portfolios):
            # 随机权重
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            results.append({
                'return': portfolio_return,
                'risk': portfolio_risk,
                'weights': weights
            })
        
        return pd.DataFrame(results)
    
    def monte_carlo_simulation(self, num_simulations=1000):
        """蒙特卡洛模拟"""
        n_assets = len(self.expected_returns)
        results = []
        
        for _ in range(num_simulations):
            # 随机权重
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # 模拟收益率
            simulated_returns = np.random.multivariate_normal(
                self.expected_returns, self.cov_matrix, size=252
            )
            
            portfolio_returns = np.sum(weights * simulated_returns, axis=1)
            portfolio_return = np.mean(portfolio_returns)
            portfolio_risk = np.std(portfolio_returns)
            
            results.append({
                'return': portfolio_return,
                'risk': portfolio_risk,
                'weights': weights
            })
        
        return pd.DataFrame(results)

def test_portfolio_optimizer():
    """测试投资组合优化器"""
    print("测试投资组合优化引擎...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n_assets = 5
    
    # 生成随机价格数据
    price_data = pd.DataFrame(
        np.random.randn(len(dates), n_assets).cumsum(axis=0) + 100,
        index=dates,
        columns=[f'Stock_{i+1}' for i in range(n_assets)]
    )
    
    # 初始化优化器
    optimizer = PortfolioOptimizer()
    optimizer.prepare_data(price_data)
    
    # 测试均值方差优化
    print("1. 均值方差优化测试...")
    mvo_result = optimizer.mean_variance_optimization()
    if mvo_result['success']:
        print(f"   预期收益率: {mvo_result['expected_return']:.4f}")
        print(f"   风险: {mvo_result['risk']:.4f}")
        print(f"   夏普比率: {mvo_result['sharpe_ratio']:.4f}")
    else:
        print(f"   优化失败: {mvo_result['message']}")
    
    # 测试风险平价优化
    print("2. 风险平价优化测试...")
    rp_result = optimizer.risk_parity_optimization()
    if rp_result['success']:
        print(f"   预期收益率: {rp_result['expected_return']:.4f}")
        print(f"   风险: {rp_result['risk']:.4f}")
    else:
        print(f"   优化失败: {rp_result['message']}")
    
    # 测试最小方差优化
    print("3. 最小方差优化测试...")
    mvo_result = optimizer.minimum_variance_optimization()
    if mvo_result['success']:
        print(f"   预期收益率: {mvo_result['expected_return']:.4f}")
        print(f"   风险: {mvo_result['risk']:.4f}")
    else:
        print(f"   优化失败: {mvo_result['message']}")
    
    # 测试有效前沿
    print("4. 有效前沿测试...")
    frontier = optimizer.efficient_frontier(50)
    print(f"   生成了 {len(frontier)} 个投资组合")
    print(f"   收益率范围: {frontier['return'].min():.4f} - {frontier['return'].max():.4f}")
    print(f"   风险范围: {frontier['risk'].min():.4f} - {frontier['risk'].max():.4f}")
    
    # 测试蒙特卡洛模拟
    print("5. 蒙特卡洛模拟测试...")
    mc_results = optimizer.monte_carlo_simulation(100)
    print(f"   模拟了 {len(mc_results)} 个场景")
    print(f"   平均收益率: {mc_results['return'].mean():.4f}")
    print(f"   平均风险: {mc_results['risk'].mean():.4f}")
    
    print("投资组合优化引擎测试完成！")
    return True

if __name__ == "__main__":
    test_portfolio_optimizer()
