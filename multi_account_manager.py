#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多账户管理系统
Multi-Account Management System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiAccountManager:
    """多账户管理器"""
    
    def __init__(self):
        self.accounts = {}
        self.users = {}
        self.roles = {}
        self.permissions = {}
        self.audit_log = []
        
        # 初始化默认角色
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """初始化默认角色"""
        self.roles = {
            'super_admin': {
                'name': '超级管理员',
                'permissions': ['*'],
                'description': '系统最高权限'
            },
            'admin': {
                'name': '管理员',
                'permissions': [
                    'account.create', 'account.read', 'account.update', 'account.delete',
                    'user.create', 'user.read', 'user.update', 'user.delete',
                    'portfolio.read', 'portfolio.update', 'trading.execute'
                ],
                'description': '系统管理员'
            },
            'portfolio_manager': {
                'name': '投资组合经理',
                'permissions': [
                    'account.read', 'portfolio.read', 'portfolio.update',
                    'trading.execute', 'analysis.read'
                ],
                'description': '投资组合管理'
            },
            'trader': {
                'name': '交易员',
                'permissions': [
                    'account.read', 'portfolio.read', 'trading.execute'
                ],
                'description': '执行交易'
            },
            'analyst': {
                'name': '分析师',
                'permissions': [
                    'account.read', 'portfolio.read', 'analysis.read', 'analysis.create'
                ],
                'description': '市场分析'
            },
            'viewer': {
                'name': '观察者',
                'permissions': [
                    'account.read', 'portfolio.read', 'analysis.read'
                ],
                'description': '只读权限'
            }
        }
    
    def create_user(self, username: str, email: str, role: str, 
                   password: str = None, **kwargs) -> str:
        """创建用户"""
        user_id = str(uuid.uuid4())
        
        if role not in self.roles:
            raise ValueError(f"无效的角色: {role}")
        
        user_data = {
            'id': user_id,
            'username': username,
            'email': email,
            'role': role,
            'password_hash': self._hash_password(password) if password else None,
            'created_at': datetime.now(),
            'last_login': None,
            'is_active': True,
            'metadata': kwargs
        }
        
        self.users[user_id] = user_data
        self._log_audit('user.create', user_id, {'username': username, 'role': role})
        
        return user_id
    
    def create_account(self, account_name: str, account_type: str, 
                      owner_id: str, initial_balance: float = 0.0,
                      **kwargs) -> str:
        """创建账户"""
        account_id = str(uuid.uuid4())
        
        if owner_id not in self.users:
            raise ValueError(f"用户不存在: {owner_id}")
        
        account_data = {
            'id': account_id,
            'name': account_name,
            'type': account_type,
            'owner_id': owner_id,
            'balance': initial_balance,
            'created_at': datetime.now(),
            'is_active': True,
            'metadata': kwargs,
            'transactions': [],
            'positions': {}
        }
        
        self.accounts[account_id] = account_data
        self._log_audit('account.create', account_id, {
            'name': account_name, 'type': account_type, 'owner_id': owner_id
        })
        
        return account_id
    
    def assign_user_to_account(self, user_id: str, account_id: str, 
                              permission_level: str = 'read') -> bool:
        """将用户分配到账户"""
        if user_id not in self.users:
            raise ValueError(f"用户不存在: {user_id}")
        
        if account_id not in self.accounts:
            raise ValueError(f"账户不存在: {account_id}")
        
        # 检查权限（管理员可以分配用户到账户）
        if not self._check_permission(user_id, 'account.update') and self.users[user_id]['role'] != 'admin':
            raise PermissionError("用户没有分配账户的权限")
        
        # 添加用户到账户
        if 'users' not in self.accounts[account_id]:
            self.accounts[account_id]['users'] = {}
        
        self.accounts[account_id]['users'][user_id] = {
            'permission_level': permission_level,
            'assigned_at': datetime.now()
        }
        
        self._log_audit('account.assign_user', account_id, {
            'user_id': user_id, 'permission_level': permission_level
        })
        
        return True
    
    def execute_trade(self, account_id: str, user_id: str, 
                     symbol: str, quantity: float, price: float,
                     trade_type: str = 'buy') -> str:
        """执行交易"""
        if not self._check_account_permission(user_id, account_id, 'trading.execute'):
            raise PermissionError("用户没有执行交易的权限")
        
        if account_id not in self.accounts:
            raise ValueError(f"账户不存在: {account_id}")
        
        # 检查账户余额
        if trade_type == 'buy':
            required_amount = quantity * price
            if self.accounts[account_id]['balance'] < required_amount:
                raise ValueError("账户余额不足")
        
        # 创建交易记录
        trade_id = str(uuid.uuid4())
        trade_data = {
            'id': trade_id,
            'account_id': account_id,
            'user_id': user_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_type': trade_type,
            'timestamp': datetime.now(),
            'status': 'executed'
        }
        
        # 更新账户
        if trade_type == 'buy':
            self.accounts[account_id]['balance'] -= quantity * price
            if symbol not in self.accounts[account_id]['positions']:
                self.accounts[account_id]['positions'][symbol] = 0
            self.accounts[account_id]['positions'][symbol] += quantity
        else:  # sell
            if symbol not in self.accounts[account_id]['positions']:
                self.accounts[account_id]['positions'][symbol] = 0
            if self.accounts[account_id]['positions'][symbol] < quantity:
                raise ValueError("持仓不足")
            self.accounts[account_id]['balance'] += quantity * price
            self.accounts[account_id]['positions'][symbol] -= quantity
        
        # 记录交易
        self.accounts[account_id]['transactions'].append(trade_data)
        
        self._log_audit('trade.execute', trade_id, {
            'account_id': account_id, 'user_id': user_id,
            'symbol': symbol, 'quantity': quantity, 'trade_type': trade_type
        })
        
        return trade_id
    
    def get_account_performance(self, account_id: str, start_date: datetime = None,
                               end_date: datetime = None) -> Dict:
        """获取账户表现"""
        if account_id not in self.accounts:
            raise ValueError(f"账户不存在: {account_id}")
        
        account = self.accounts[account_id]
        transactions = account['transactions']
        
        if start_date:
            transactions = [t for t in transactions if t['timestamp'] >= start_date]
        if end_date:
            transactions = [t for t in transactions if t['timestamp'] <= end_date]
        
        # 计算表现指标
        total_trades = len(transactions)
        buy_trades = len([t for t in transactions if t['trade_type'] == 'buy'])
        sell_trades = len([t for t in transactions if t['trade_type'] == 'sell'])
        
        total_volume = sum(t['quantity'] * t['price'] for t in transactions)
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        
        # 计算持仓价值
        current_positions = account['positions']
        position_value = sum(
            pos * 100 for pos in current_positions.values()  # 假设当前价格为100
        )
        
        return {
            'account_id': account_id,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_volume': total_volume,
            'avg_trade_size': avg_trade_size,
            'current_balance': account['balance'],
            'position_value': position_value,
            'total_value': account['balance'] + position_value
        }
    
    def get_user_accounts(self, user_id: str) -> List[Dict]:
        """获取用户可访问的账户"""
        if user_id not in self.users:
            raise ValueError(f"用户不存在: {user_id}")
        
        user_accounts = []
        for account_id, account in self.accounts.items():
            # 检查用户是否是账户所有者
            if account['owner_id'] == user_id:
                user_accounts.append({
                    'account_id': account_id,
                    'name': account['name'],
                    'type': account['type'],
                    'permission_level': 'owner',
                    'balance': account['balance']
                })
            # 检查用户是否被分配到账户
            elif 'users' in account and user_id in account['users']:
                user_accounts.append({
                    'account_id': account_id,
                    'name': account['name'],
                    'type': account['type'],
                    'permission_level': account['users'][user_id]['permission_level'],
                    'balance': account['balance']
                })
        
        return user_accounts
    
    def get_audit_log(self, user_id: str = None, action: str = None,
                     start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """获取审计日志"""
        # 如果没有指定用户ID，或者用户没有权限，返回空列表
        if user_id and not self._check_permission(user_id, 'audit.read'):
            return []
        
        filtered_log = self.audit_log
        
        if user_id:
            filtered_log = [log for log in filtered_log if log.get('user_id') == user_id]
        if action:
            filtered_log = [log for log in filtered_log if log.get('action') == action]
        if start_date:
            filtered_log = [log for log in filtered_log if log.get('timestamp') >= start_date]
        if end_date:
            filtered_log = [log for log in filtered_log if log.get('timestamp') <= end_date]
        
        return filtered_log
    
    def _check_permission(self, user_id: str, permission: str) -> bool:
        """检查用户权限"""
        if user_id not in self.users:
            return False
        
        user_role = self.users[user_id]['role']
        role_permissions = self.roles[user_role]['permissions']
        
        # 检查通配符权限
        if '*' in role_permissions:
            return True
        
        # 检查具体权限
        return permission in role_permissions
    
    def _check_account_permission(self, user_id: str, account_id: str, 
                                permission: str) -> bool:
        """检查账户权限"""
        if not self._check_permission(user_id, permission):
            return False
        
        # 检查用户是否是账户所有者
        if self.accounts[account_id]['owner_id'] == user_id:
            return True
        
        # 检查用户是否被分配到账户
        if 'users' in self.accounts[account_id] and user_id in self.accounts[account_id]['users']:
            user_permission = self.accounts[account_id]['users'][user_id]['permission_level']
            return user_permission in ['read', 'write', 'execute']
        
        return False
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _log_audit(self, action: str, resource_id: str, details: Dict):
        """记录审计日志"""
        log_entry = {
            'id': str(uuid.uuid4()),
            'action': action,
            'resource_id': resource_id,
            'details': details,
            'timestamp': datetime.now()
        }
        self.audit_log.append(log_entry)

def test_multi_account_manager():
    """测试多账户管理系统"""
    print("测试多账户管理系统...")
    
    # 初始化管理器
    manager = MultiAccountManager()
    
    # 创建用户
    print("1. 创建用户...")
    admin_id = manager.create_user('admin', 'admin@example.com', 'admin', 'password123')
    trader_id = manager.create_user('trader', 'trader@example.com', 'trader', 'password123')
    analyst_id = manager.create_user('analyst', 'analyst@example.com', 'analyst', 'password123')
    
    print(f"   创建了 {len(manager.users)} 个用户")
    
    # 创建账户
    print("2. 创建账户...")
    account1_id = manager.create_account('主账户', 'trading', admin_id, 100000.0)
    account2_id = manager.create_account('测试账户', 'testing', admin_id, 50000.0)
    
    print(f"   创建了 {len(manager.accounts)} 个账户")
    
    # 分配用户到账户
    print("3. 分配用户到账户...")
    manager.assign_user_to_account(admin_id, account1_id, 'execute')  # 管理员分配
    manager.assign_user_to_account(admin_id, account2_id, 'read')     # 管理员分配
    
    print("   用户分配完成")
    
    # 执行交易
    print("4. 执行交易...")
    try:
        trade1_id = manager.execute_trade(account1_id, admin_id, 'AAPL', 100, 150.0, 'buy')
        print(f"   买入交易: {trade1_id}")
        
        trade2_id = manager.execute_trade(account1_id, admin_id, 'MSFT', 50, 300.0, 'buy')
        print(f"   买入交易: {trade2_id}")
        
        trade3_id = manager.execute_trade(account1_id, admin_id, 'AAPL', 50, 155.0, 'sell')
        print(f"   卖出交易: {trade3_id}")
    except Exception as e:
        print(f"   交易执行失败: {e}")
    
    # 获取账户表现
    print("5. 获取账户表现...")
    performance = manager.get_account_performance(account1_id)
    print(f"   总交易数: {performance['total_trades']}")
    print(f"   买入交易: {performance['buy_trades']}")
    print(f"   卖出交易: {performance['sell_trades']}")
    print(f"   当前余额: {performance['current_balance']:.2f}")
    print(f"   持仓价值: {performance['position_value']:.2f}")
    
    # 获取用户账户
    print("6. 获取用户账户...")
    admin_accounts = manager.get_user_accounts(admin_id)
    print(f"   管理员拥有 {len(admin_accounts)} 个账户")
    
    trader_accounts = manager.get_user_accounts(trader_id)
    print(f"   交易员可访问 {len(trader_accounts)} 个账户")
    
    # 获取审计日志
    print("7. 获取审计日志...")
    audit_log = manager.get_audit_log()
    print(f"   审计日志条目: {len(audit_log)}")
    
    # 测试权限检查
    print("8. 测试权限检查...")
    try:
        # 分析师尝试执行交易（应该失败）
        manager.execute_trade(account1_id, analyst_id, 'GOOGL', 10, 2500.0, 'buy')
        print("   权限检查失败：分析师不应该能执行交易")
    except PermissionError:
        print("   权限检查通过：分析师无法执行交易")
    
    print("多账户管理系统测试完成！")
    return True

if __name__ == "__main__":
    test_multi_account_manager()
