#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监管合规工具
Regulatory Compliance Tools
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import uuid
import warnings
warnings.filterwarnings('ignore')

class ComplianceManager:
    """合规管理器"""
    
    def __init__(self):
        self.rules = {}
        self.violations = []
        self.reports = {}
        self.audit_trail = []
        
        # 初始化默认合规规则
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认合规规则"""
        self.rules = {
            'concentration_limit': {
                'name': '集中度限制',
                'type': 'concentration',
                'max_weight': 0.1,  # 单个资产最大权重10%
                'description': '单个资产权重不能超过10%'
            },
            'sector_limit': {
                'name': '行业集中度限制',
                'type': 'sector',
                'max_sector_weight': 0.3,  # 单个行业最大权重30%
                'description': '单个行业权重不能超过30%'
            },
            'var_limit': {
                'name': '风险价值限制',
                'type': 'var',
                'max_var': 0.05,  # 最大VaR 5%
                'confidence': 0.95,
                'description': '95%置信度下VaR不能超过5%'
            },
            'turnover_limit': {
                'name': '换手率限制',
                'type': 'turnover',
                'max_turnover': 2.0,  # 最大年化换手率200%
                'description': '年化换手率不能超过200%'
            },
            'position_size_limit': {
                'name': '持仓规模限制',
                'type': 'position_size',
                'max_position_value': 1000000,  # 单个持仓最大价值100万
                'description': '单个持仓价值不能超过100万'
            },
            'leverage_limit': {
                'name': '杠杆限制',
                'type': 'leverage',
                'max_leverage': 2.0,  # 最大杠杆2倍
                'description': '总杠杆不能超过2倍'
            }
        }
    
    def add_rule(self, rule_id: str, rule_config: Dict) -> bool:
        """添加合规规则"""
        self.rules[rule_id] = rule_config
        self._log_audit('rule.add', rule_id, rule_config)
        return True
    
    def update_rule(self, rule_id: str, rule_config: Dict) -> bool:
        """更新合规规则"""
        if rule_id not in self.rules:
            raise ValueError(f"规则不存在: {rule_id}")
        
        self.rules[rule_id] = rule_config
        self._log_audit('rule.update', rule_id, rule_config)
        return True
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除合规规则"""
        if rule_id not in self.rules:
            raise ValueError(f"规则不存在: {rule_id}")
        
        del self.rules[rule_id]
        self._log_audit('rule.delete', rule_id, {})
        return True
    
    def check_concentration_limit(self, portfolio: Dict, rule_config: Dict) -> Dict:
        """检查集中度限制"""
        violations = []
        max_weight = rule_config['max_weight']
        
        for asset, weight in portfolio.items():
            if weight > max_weight:
                violations.append({
                    'asset': asset,
                    'current_weight': weight,
                    'max_weight': max_weight,
                    'violation_amount': weight - max_weight,
                    'severity': 'high' if weight > max_weight * 1.5 else 'medium'
                })
        
        return {
            'rule_id': 'concentration_limit',
            'rule_name': rule_config.get('name', '集中度限制'),
            'violations': violations,
            'compliant': len(violations) == 0,
            'severity': 'high' if any(v['severity'] == 'high' for v in violations) else 'medium' if violations else 'low'
        }
    
    def check_sector_limit(self, portfolio: Dict, sector_mapping: Dict, rule_config: Dict) -> Dict:
        """检查行业集中度限制"""
        violations = []
        max_sector_weight = rule_config['max_sector_weight']
        
        # 计算各行业权重
        sector_weights = {}
        for asset, weight in portfolio.items():
            if asset in sector_mapping:
                sector = sector_mapping[asset]
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weight
        
        # 检查行业权重限制
        for sector, weight in sector_weights.items():
            if weight > max_sector_weight:
                violations.append({
                    'sector': sector,
                    'current_weight': weight,
                    'max_weight': max_sector_weight,
                    'violation_amount': weight - max_sector_weight,
                    'severity': 'high' if weight > max_sector_weight * 1.5 else 'medium'
                })
        
        return {
            'rule_id': 'sector_limit',
            'rule_name': rule_config.get('name', '行业集中度限制'),
            'violations': violations,
            'sector_weights': sector_weights,
            'compliant': len(violations) == 0,
            'severity': 'high' if any(v['severity'] == 'high' for v in violations) else 'medium' if violations else 'low'
        }
    
    def check_var_limit(self, portfolio_returns: pd.Series, rule_config: Dict) -> Dict:
        """检查VaR限制"""
        max_var = rule_config['max_var']
        confidence = rule_config['confidence']
        
        # 计算VaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        var_abs = abs(var)
        
        violation = None
        if var_abs > max_var:
            violation = {
                'current_var': var_abs,
                'max_var': max_var,
                'violation_amount': var_abs - max_var,
                'severity': 'high' if var_abs > max_var * 1.5 else 'medium'
            }
        
        return {
            'rule_id': 'var_limit',
            'rule_name': rule_config.get('name', '风险价值限制'),
            'var': var,
            'var_abs': var_abs,
            'max_var': max_var,
            'violation': violation,
            'compliant': violation is None,
            'severity': violation['severity'] if violation else 'low'
        }
    
    def check_turnover_limit(self, transactions: List[Dict], rule_config: Dict) -> Dict:
        """检查换手率限制"""
        max_turnover = rule_config['max_turnover']
        
        if not transactions:
            return {
                'rule_id': 'turnover_limit',
                'rule_name': rule_config.get('name', '换手率限制'),
                'turnover': 0,
                'max_turnover': max_turnover,
                'compliant': True,
                'severity': 'low'
            }
        
        # 计算换手率
        total_volume = sum(t['quantity'] * t['price'] for t in transactions)
        # 假设投资组合价值为100万（实际应该从账户数据获取）
        portfolio_value = 1000000
        turnover = total_volume / portfolio_value
        
        violation = None
        if turnover > max_turnover:
            violation = {
                'current_turnover': turnover,
                'max_turnover': max_turnover,
                'violation_amount': turnover - max_turnover,
                'severity': 'high' if turnover > max_turnover * 1.5 else 'medium'
            }
        
        return {
            'rule_id': 'turnover_limit',
            'rule_name': rule_config.get('name', '换手率限制'),
            'turnover': turnover,
            'max_turnover': max_turnover,
            'violation': violation,
            'compliant': violation is None,
            'severity': violation['severity'] if violation else 'low'
        }
    
    def check_position_size_limit(self, positions: Dict, prices: Dict, rule_config: Dict) -> Dict:
        """检查持仓规模限制"""
        max_position_value = rule_config['max_position_value']
        violations = []
        
        for asset, quantity in positions.items():
            if asset in prices:
                position_value = quantity * prices[asset]
                if position_value > max_position_value:
                    violations.append({
                        'asset': asset,
                        'current_value': position_value,
                        'max_value': max_position_value,
                        'violation_amount': position_value - max_position_value,
                        'severity': 'high' if position_value > max_position_value * 1.5 else 'medium'
                    })
        
        return {
            'rule_id': 'position_size_limit',
            'rule_name': rule_config.get('name', '持仓规模限制'),
            'violations': violations,
            'compliant': len(violations) == 0,
            'severity': 'high' if any(v['severity'] == 'high' for v in violations) else 'medium' if violations else 'low'
        }
    
    def check_leverage_limit(self, portfolio: Dict, leverage_data: Dict, rule_config: Dict) -> Dict:
        """检查杠杆限制"""
        max_leverage = rule_config['max_leverage']
        
        # 计算总杠杆
        total_leverage = sum(leverage_data.values())
        
        violation = None
        if total_leverage > max_leverage:
            violation = {
                'current_leverage': total_leverage,
                'max_leverage': max_leverage,
                'violation_amount': total_leverage - max_leverage,
                'severity': 'high' if total_leverage > max_leverage * 1.5 else 'medium'
            }
        
        return {
            'rule_id': 'leverage_limit',
            'rule_name': rule_config.get('name', '杠杆限制'),
            'leverage': total_leverage,
            'max_leverage': max_leverage,
            'violation': violation,
            'compliant': violation is None,
            'severity': violation['severity'] if violation else 'low'
        }
    
    def comprehensive_compliance_check(self, portfolio_data: Dict) -> Dict:
        """综合合规检查"""
        results = {}
        overall_compliant = True
        high_severity_violations = 0
        
        # 检查集中度限制
        if 'concentration_limit' in self.rules:
            concentration_result = self.check_concentration_limit(
                portfolio_data['portfolio'], 
                self.rules['concentration_limit']
            )
            results['concentration'] = concentration_result
            if not concentration_result['compliant']:
                overall_compliant = False
                if concentration_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 检查行业集中度限制
        if 'sector_limit' in self.rules and 'sector_mapping' in portfolio_data:
            sector_result = self.check_sector_limit(
                portfolio_data['portfolio'],
                portfolio_data['sector_mapping'],
                self.rules['sector_limit']
            )
            results['sector'] = sector_result
            if not sector_result['compliant']:
                overall_compliant = False
                if sector_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 检查VaR限制
        if 'var_limit' in self.rules and 'portfolio_returns' in portfolio_data:
            var_result = self.check_var_limit(
                portfolio_data['portfolio_returns'],
                self.rules['var_limit']
            )
            results['var'] = var_result
            if not var_result['compliant']:
                overall_compliant = False
                if var_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 检查换手率限制
        if 'turnover_limit' in self.rules and 'transactions' in portfolio_data:
            turnover_result = self.check_turnover_limit(
                portfolio_data['transactions'],
                self.rules['turnover_limit']
            )
            results['turnover'] = turnover_result
            if not turnover_result['compliant']:
                overall_compliant = False
                if turnover_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 检查持仓规模限制
        if 'position_size_limit' in self.rules and 'positions' in portfolio_data and 'prices' in portfolio_data:
            position_result = self.check_position_size_limit(
                portfolio_data['positions'],
                portfolio_data['prices'],
                self.rules['position_size_limit']
            )
            results['position_size'] = position_result
            if not position_result['compliant']:
                overall_compliant = False
                if position_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 检查杠杆限制
        if 'leverage_limit' in self.rules and 'leverage_data' in portfolio_data:
            leverage_result = self.check_leverage_limit(
                portfolio_data['portfolio'],
                portfolio_data['leverage_data'],
                self.rules['leverage_limit']
            )
            results['leverage'] = leverage_result
            if not leverage_result['compliant']:
                overall_compliant = False
                if leverage_result['severity'] == 'high':
                    high_severity_violations += 1
        
        # 生成综合报告
        summary = {
            'overall_compliant': overall_compliant,
            'high_severity_violations': high_severity_violations,
            'total_rules_checked': len(results),
            'compliant_rules': sum(1 for r in results.values() if r['compliant']),
            'violated_rules': sum(1 for r in results.values() if not r['compliant']),
            'timestamp': datetime.now()
        }
        
        return {
            'summary': summary,
            'results': results,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成合规建议"""
        recommendations = []
        
        for rule_name, result in results.items():
            if not result['compliant']:
                if rule_name == 'concentration':
                    recommendations.append("建议降低单个资产权重，分散投资风险")
                elif rule_name == 'sector':
                    recommendations.append("建议降低行业集中度，增加行业多样性")
                elif rule_name == 'var':
                    recommendations.append("建议降低投资组合风险，减少高风险资产配置")
                elif rule_name == 'turnover':
                    recommendations.append("建议降低交易频率，减少换手率")
                elif rule_name == 'position_size':
                    recommendations.append("建议减少单个持仓规模，分散投资")
                elif rule_name == 'leverage':
                    recommendations.append("建议降低杠杆水平，控制风险敞口")
        
        return recommendations
    
    def generate_compliance_report(self, compliance_results: Dict, format: str = 'json') -> str:
        """生成合规报告"""
        report_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        report = {
            'report_id': report_id,
            'timestamp': timestamp,
            'summary': compliance_results['summary'],
            'detailed_results': compliance_results['results'],
            'recommendations': compliance_results['recommendations']
        }
        
        self.reports[report_id] = report
        
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'html':
            return self._generate_html_report(report)
        else:
            return str(report)
    
    def _generate_html_report(self, report: Dict) -> str:
        """生成HTML格式报告"""
        html = f"""
        <html>
        <head>
            <title>合规检查报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .violation {{ background-color: #ffebee; padding: 10px; margin: 10px 0; border-left: 4px solid #f44336; }}
                .compliant {{ background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50; }}
                .recommendations {{ background-color: #fff3e0; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>合规检查报告</h1>
                <p>报告ID: {report['report_id']}</p>
                <p>生成时间: {report['timestamp']}</p>
            </div>
            
            <div class="summary">
                <h2>检查摘要</h2>
                <p>总体合规状态: {'合规' if report['summary']['overall_compliant'] else '不合规'}</p>
                <p>高风险违规数量: {report['summary']['high_severity_violations']}</p>
                <p>检查规则数量: {report['summary']['total_rules_checked']}</p>
                <p>合规规则数量: {report['summary']['compliant_rules']}</p>
                <p>违规规则数量: {report['summary']['violated_rules']}</p>
            </div>
        """
        
        # 添加详细结果
        for rule_name, result in report['detailed_results'].items():
            status_class = 'compliant' if result['compliant'] else 'violation'
            html += f"""
            <div class="{status_class}">
                <h3>{result['rule_name']}</h3>
                <p>状态: {'合规' if result['compliant'] else '不合规'}</p>
                <p>严重程度: {result['severity']}</p>
            </div>
            """
        
        # 添加建议
        if report['recommendations']:
            html += """
            <div class="recommendations">
                <h2>合规建议</h2>
                <ul>
            """
            for rec in report['recommendations']:
                html += f"<li>{rec}</li>"
            html += """
                </ul>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _log_audit(self, action: str, resource_id: str, details: Dict):
        """记录审计日志"""
        log_entry = {
            'id': str(uuid.uuid4()),
            'action': action,
            'resource_id': resource_id,
            'details': details,
            'timestamp': datetime.now()
        }
        self.audit_trail.append(log_entry)

def test_compliance_tools():
    """测试合规工具"""
    print("测试合规工具...")
    
    # 初始化合规管理器
    compliance = ComplianceManager()
    
    # 创建测试数据
    portfolio = {
        'AAPL': 0.15,  # 15% - 超过10%限制
        'MSFT': 0.12,  # 12% - 超过10%限制
        'GOOGL': 0.08,
        'AMZN': 0.07,
        'TSLA': 0.06,
        'META': 0.05,
        'NVDA': 0.04,
        'NFLX': 0.03,
        '其他': 0.40
    }
    
    sector_mapping = {
        'AAPL': '科技',
        'MSFT': '科技',
        'GOOGL': '科技',
        'AMZN': '科技',
        'TSLA': '汽车',
        'META': '科技',
        'NVDA': '科技',
        'NFLX': '媒体',
        '其他': '其他'
    }
    
    # 生成模拟收益率数据
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)  # 一年的日收益率
    
    # 创建测试交易记录
    transactions = [
        {'quantity': 100, 'price': 150.0},
        {'quantity': 50, 'price': 300.0},
        {'quantity': 200, 'price': 2500.0}
    ]
    
    positions = {'AAPL': 100, 'MSFT': 50, 'GOOGL': 10}
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
    leverage_data = {'AAPL': 1.2, 'MSFT': 1.1, 'GOOGL': 1.0}
    
    portfolio_data = {
        'portfolio': portfolio,
        'sector_mapping': sector_mapping,
        'portfolio_returns': returns,
        'transactions': transactions,
        'positions': positions,
        'prices': prices,
        'leverage_data': leverage_data
    }
    
    # 测试集中度检查
    print("1. 测试集中度检查...")
    concentration_result = compliance.check_concentration_limit(portfolio, compliance.rules['concentration_limit'])
    print(f"   合规状态: {'合规' if concentration_result['compliant'] else '不合规'}")
    print(f"   违规数量: {len(concentration_result['violations'])}")
    
    # 测试行业集中度检查
    print("2. 测试行业集中度检查...")
    sector_result = compliance.check_sector_limit(portfolio, sector_mapping, compliance.rules['sector_limit'])
    print(f"   合规状态: {'合规' if sector_result['compliant'] else '不合规'}")
    print(f"   违规数量: {len(sector_result['violations'])}")
    
    # 测试VaR检查
    print("3. 测试VaR检查...")
    var_result = compliance.check_var_limit(returns, compliance.rules['var_limit'])
    print(f"   合规状态: {'合规' if var_result['compliant'] else '不合规'}")
    print(f"   当前VaR: {var_result['var_abs']:.4f}")
    print(f"   最大VaR: {var_result['max_var']:.4f}")
    
    # 测试换手率检查
    print("4. 测试换手率检查...")
    turnover_result = compliance.check_turnover_limit(transactions, compliance.rules['turnover_limit'])
    print(f"   合规状态: {'合规' if turnover_result['compliant'] else '不合规'}")
    print(f"   当前换手率: {turnover_result['turnover']:.4f}")
    print(f"   最大换手率: {turnover_result['max_turnover']:.4f}")
    
    # 测试持仓规模检查
    print("5. 测试持仓规模检查...")
    position_result = compliance.check_position_size_limit(positions, prices, compliance.rules['position_size_limit'])
    print(f"   合规状态: {'合规' if position_result['compliant'] else '不合规'}")
    print(f"   违规数量: {len(position_result['violations'])}")
    
    # 测试杠杆检查
    print("6. 测试杠杆检查...")
    leverage_result = compliance.check_leverage_limit(portfolio, leverage_data, compliance.rules['leverage_limit'])
    print(f"   合规状态: {'合规' if leverage_result['compliant'] else '不合规'}")
    print(f"   当前杠杆: {leverage_result['leverage']:.4f}")
    print(f"   最大杠杆: {leverage_result['max_leverage']:.4f}")
    
    # 测试综合合规检查
    print("7. 测试综合合规检查...")
    comprehensive_result = compliance.comprehensive_compliance_check(portfolio_data)
    print(f"   总体合规状态: {'合规' if comprehensive_result['summary']['overall_compliant'] else '不合规'}")
    print(f"   高风险违规数量: {comprehensive_result['summary']['high_severity_violations']}")
    print(f"   检查规则数量: {comprehensive_result['summary']['total_rules_checked']}")
    print(f"   合规规则数量: {comprehensive_result['summary']['compliant_rules']}")
    print(f"   违规规则数量: {comprehensive_result['summary']['violated_rules']}")
    
    # 测试报告生成
    print("8. 测试报告生成...")
    json_report = compliance.generate_compliance_report(comprehensive_result, 'json')
    print(f"   JSON报告长度: {len(json_report)} 字符")
    
    html_report = compliance.generate_compliance_report(comprehensive_result, 'html')
    print(f"   HTML报告长度: {len(html_report)} 字符")
    
    print("合规工具测试完成！")
    return True

if __name__ == "__main__":
    test_compliance_tools()
