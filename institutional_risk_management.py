#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机构级风险管理
Institutional Risk Management
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class InstitutionalRiskManager:
    """机构级风险管理器"""
    
    def __init__(self):
        self.portfolio = None
        self.risk_limits = {}
        self.var_confidence = 0.95
        self.cvar_confidence = 0.95
        
    def set_risk_limits(self, limits):
        """设置风险限制"""
        self.risk_limits = limits
    
    def calculate_var(self, returns, confidence=0.95):
        """计算风险价值(VaR)"""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        # 历史模拟法
        var_historical = np.percentile(returns, (1 - confidence) * 100)
        
        # 参数法（假设正态分布）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        var_parametric = mean_return + stats.norm.ppf(1 - confidence) * std_return
        
        # 蒙特卡洛模拟
        n_simulations = 10000
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        var_monte_carlo = np.percentile(simulated_returns, (1 - confidence) * 100)
        
        return {
            'historical': var_historical,
            'parametric': var_parametric,
            'monte_carlo': var_monte_carlo,
            'confidence': confidence
        }
    
    def calculate_cvar(self, returns, confidence=0.95):
        """计算条件风险价值(CVaR)"""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        
        return {
            'cvar': cvar,
            'var': var,
            'confidence': confidence
        }
    
    def stress_testing(self, portfolio_returns, scenarios):
        """压力测试"""
        results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            # 计算场景下的投资组合表现
            scenario_returns = portfolio_returns * scenario_data['factor']
            scenario_loss = -np.sum(scenario_returns)
            
            results[scenario_name] = {
                'loss': scenario_loss,
                'return': np.mean(scenario_returns),
                'volatility': np.std(scenario_returns),
                'max_drawdown': self.calculate_max_drawdown(scenario_returns)
            }
        
        return results
    
    def scenario_analysis(self, portfolio_returns, scenarios):
        """情景分析"""
        results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            # 应用情景因子
            adjusted_returns = portfolio_returns.copy()
            for asset, factor in scenario_data['factors'].items():
                if asset in adjusted_returns.columns:
                    adjusted_returns[asset] *= factor
            
            # 计算情景结果
            portfolio_return = np.sum(adjusted_returns.mean() * scenario_data.get('weights', {}))
            portfolio_volatility = np.sqrt(np.dot(
                scenario_data.get('weights', {}).values(),
                np.dot(adjusted_returns.cov(), scenario_data.get('weights', {}).values())
            ))
            
            results[scenario_name] = {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            }
        
        return results
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def risk_attribution(self, portfolio_returns, factor_returns):
        """风险归因分析"""
        # 计算因子暴露
        factor_exposures = {}
        for factor, factor_returns in factor_returns.items():
            correlation = portfolio_returns.corr(factor_returns)
            factor_exposures[factor] = correlation
        
        # 计算风险贡献
        portfolio_variance = np.var(portfolio_returns)
        risk_contributions = {}
        
        for factor, exposure in factor_exposures.items():
            factor_variance = np.var(factor_returns[factor])
            risk_contributions[factor] = exposure * factor_variance / portfolio_variance
        
        return {
            'factor_exposures': factor_exposures,
            'risk_contributions': risk_contributions,
            'portfolio_variance': portfolio_variance
        }
    
    def compliance_check(self, portfolio, rules):
        """合规检查"""
        violations = []
        
        for rule_name, rule_config in rules.items():
            if rule_config['type'] == 'concentration':
                # 集中度检查
                max_weight = rule_config['max_weight']
                for asset, weight in portfolio.items():
                    if weight > max_weight:
                        violations.append({
                            'rule': rule_name,
                            'asset': asset,
                            'current_weight': weight,
                            'max_weight': max_weight,
                            'violation': True
                        })
            
            elif rule_config['type'] == 'sector_limit':
                # 行业限制检查
                sector_weights = rule_config.get('sector_weights', {})
                max_sector_weight = rule_config['max_sector_weight']
                
                for sector, weight in sector_weights.items():
                    if weight > max_sector_weight:
                        violations.append({
                            'rule': rule_name,
                            'sector': sector,
                            'current_weight': weight,
                            'max_weight': max_sector_weight,
                            'violation': True
                        })
            
            elif rule_config['type'] == 'var_limit':
                # VaR限制检查
                portfolio_returns = rule_config.get('portfolio_returns')
                if portfolio_returns is not None:
                    var_result = self.calculate_var(portfolio_returns, rule_config['confidence'])
                    max_var = rule_config['max_var']
                    
                    if abs(var_result['historical']) > max_var:
                        violations.append({
                            'rule': rule_name,
                            'current_var': abs(var_result['historical']),
                            'max_var': max_var,
                            'violation': True
                        })
        
        return {
            'violations': violations,
            'compliant': len(violations) == 0,
            'total_violations': len(violations)
        }
    
    def risk_budgeting(self, target_risk, asset_returns):
        """风险预算"""
        n_assets = len(asset_returns.columns)
        cov_matrix = asset_returns.cov()
        
        # 等权重风险预算
        equal_weights = np.ones(n_assets) / n_assets
        equal_risk_contrib = np.sqrt(np.diag(cov_matrix)) / np.sum(np.sqrt(np.diag(cov_matrix)))
        
        # 风险平价
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            risk_contrib = weights * np.dot(cov_matrix, weights) / portfolio_vol
            return np.var(risk_contrib)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            risk_parity_weights = result.x
        else:
            risk_parity_weights = equal_weights
        
        return {
            'equal_weights': equal_weights,
            'equal_risk_contrib': equal_risk_contrib,
            'risk_parity_weights': risk_parity_weights,
            'target_risk': target_risk
        }
    
    def liquidity_risk_assessment(self, portfolio, market_data):
        """流动性风险评估"""
        liquidity_metrics = {}
        
        for asset, weight in portfolio.items():
            if asset in market_data:
                # 计算流动性指标
                volume = market_data[asset].get('volume', 0)
                price = market_data[asset].get('price', 1)
                
                # 流动性比率
                liquidity_ratio = volume / (weight * 1000000)  # 假设投资组合价值100万
                
                # 流动性风险等级
                if liquidity_ratio > 10:
                    risk_level = 'Low'
                elif liquidity_ratio > 5:
                    risk_level = 'Medium'
                else:
                    risk_level = 'High'
                
                liquidity_metrics[asset] = {
                    'liquidity_ratio': liquidity_ratio,
                    'risk_level': risk_level,
                    'volume': volume,
                    'weight': weight
                }
        
        return liquidity_metrics
    
    def comprehensive_risk_report(self, portfolio, returns_data, market_data):
        """综合风险报告"""
        report = {}
        
        # VaR分析
        portfolio_returns = returns_data.mean(axis=1)
        var_results = self.calculate_var(portfolio_returns)
        cvar_results = self.calculate_cvar(portfolio_returns)
        
        report['var_analysis'] = {
            'var': var_results,
            'cvar': cvar_results
        }
        
        # 压力测试
        stress_scenarios = {
            'market_crash': {'factor': -0.2},
            'interest_rate_shock': {'factor': -0.1},
            'currency_crisis': {'factor': -0.15}
        }
        stress_results = self.stress_testing(portfolio_returns, stress_scenarios)
        report['stress_testing'] = stress_results
        
        # 最大回撤
        max_dd = self.calculate_max_drawdown(portfolio_returns)
        report['max_drawdown'] = max_dd
        
        # 流动性风险
        liquidity_risk = self.liquidity_risk_assessment(portfolio, market_data)
        report['liquidity_risk'] = liquidity_risk
        
        # 合规检查
        compliance_rules = {
            'concentration_limit': {
                'type': 'concentration',
                'max_weight': 0.1
            },
            'var_limit': {
                'type': 'var_limit',
                'max_var': 0.05,
                'confidence': 0.95,
                'portfolio_returns': portfolio_returns
            }
        }
        compliance_results = self.compliance_check(portfolio, compliance_rules)
        report['compliance'] = compliance_results
        
        return report

def test_institutional_risk_management():
    """测试机构级风险管理"""
    print("测试机构级风险管理...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n_assets = 5
    
    # 生成资产收益率数据
    returns_data = pd.DataFrame(
        np.random.randn(len(dates), n_assets) * 0.02,
        index=dates,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # 生成市场数据
    market_data = {}
    for i in range(n_assets):
        market_data[f'Asset_{i+1}'] = {
            'price': 100 + np.random.randn() * 10,
            'volume': np.random.randint(1000, 10000)
        }
    
    # 创建投资组合
    portfolio = {f'Asset_{i+1}': 0.2 for i in range(n_assets)}
    
    # 初始化风险管理器
    risk_manager = InstitutionalRiskManager()
    
    # 测试VaR计算
    print("1. 测试VaR计算...")
    portfolio_returns = returns_data.mean(axis=1)
    var_results = risk_manager.calculate_var(portfolio_returns)
    print(f"   历史VaR: {var_results['historical']:.4f}")
    print(f"   参数VaR: {var_results['parametric']:.4f}")
    print(f"   蒙特卡洛VaR: {var_results['monte_carlo']:.4f}")
    
    # 测试CVaR计算
    print("2. 测试CVaR计算...")
    cvar_results = risk_manager.calculate_cvar(portfolio_returns)
    print(f"   CVaR: {cvar_results['cvar']:.4f}")
    print(f"   VaR: {cvar_results['var']:.4f}")
    
    # 测试压力测试
    print("3. 测试压力测试...")
    stress_scenarios = {
        'market_crash': {'factor': -0.2},
        'interest_rate_shock': {'factor': -0.1}
    }
    stress_results = risk_manager.stress_testing(portfolio_returns, stress_scenarios)
    for scenario, result in stress_results.items():
        print(f"   {scenario}: 损失 {result['loss']:.4f}")
    
    # 测试最大回撤
    print("4. 测试最大回撤...")
    max_dd = risk_manager.calculate_max_drawdown(portfolio_returns)
    print(f"   最大回撤: {max_dd:.4f}")
    
    # 测试合规检查
    print("5. 测试合规检查...")
    compliance_rules = {
        'concentration_limit': {
            'type': 'concentration',
            'max_weight': 0.1
        }
    }
    compliance_results = risk_manager.compliance_check(portfolio, compliance_rules)
    print(f"   合规状态: {'通过' if compliance_results['compliant'] else '违规'}")
    print(f"   违规数量: {compliance_results['total_violations']}")
    
    # 测试流动性风险评估
    print("6. 测试流动性风险评估...")
    liquidity_risk = risk_manager.liquidity_risk_assessment(portfolio, market_data)
    for asset, metrics in liquidity_risk.items():
        print(f"   {asset}: 风险等级 {metrics['risk_level']}")
    
    # 测试综合风险报告
    print("7. 测试综合风险报告...")
    comprehensive_report = risk_manager.comprehensive_risk_report(
        portfolio, returns_data, market_data
    )
    print(f"   报告包含 {len(comprehensive_report)} 个风险指标")
    
    print("机构级风险管理测试完成！")
    return True

if __name__ == "__main__":
    test_institutional_risk_management()
