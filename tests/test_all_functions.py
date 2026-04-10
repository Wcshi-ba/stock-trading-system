#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面测试所有功能与后端的连接状态
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(method, endpoint, data=None, description=""):
    """测试单个端点"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"\n{'='*60}")
        print(f"测试: {description}")
        print(f"请求: {method} {endpoint}")
        
        if method.upper() == 'GET':
            response = requests.get(url, params=data)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data)
        else:
            print(f"不支持的HTTP方法: {method}")
            return False
            
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"成功: {result.get('message', 'OK')}")
                if 'success' in result:
                    print(f"   成功标志: {result['success']}")
                return True
            except:
                print(f"成功: {response.text[:100]}...")
                return True
        else:
            print(f"失败: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"异常: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始全面测试所有功能与后端的连接状态")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试结果统计
    total_tests = 0
    passed_tests = 0
    
    # 1. 基础功能测试
    print("\n" + "="*80)
    print("基础功能测试")
    print("="*80)
    
    tests = [
        # 健康检查
        ("GET", "/", None, "首页访问"),
        ("GET", "/api/health", None, "健康检查"),
        
        # 数据获取
        ("GET", "/get_data", {"ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2024-12-31"}, "获取AAPL数据"),
        ("GET", "/get_data", {"ticker": "600519", "start_date": "2020-01-01", "end_date": "2024-12-31"}, "获取A股数据"),
        
        # 模型训练
        ("POST", "/train_model", {"ticker": "AAPL", "epochs": 5, "learning_rate": 0.001, "initial_money": 10000}, "训练AAPL模型"),
        
        # 静态文件服务
        ("GET", "/tmp/flask/ticker/AAPL.csv", None, "原始数据文件"),
        ("GET", "/results/transactions/AAPL_transactions.csv", None, "交易记录文件"),
        ("GET", "/images/predictions/AAPL_prediction.png", None, "预测图片"),
    ]
    
    for method, endpoint, data, desc in tests:
        total_tests += 1
        if test_endpoint(method, endpoint, data, desc):
            passed_tests += 1
        time.sleep(0.5)  # 避免请求过快
    
    # 2. 高级功能测试
    print("\n" + "="*80)
    print("高级功能测试")
    print("="*80)
    
    advanced_tests = [
        # 多策略回测
        ("POST", "/api/strategies/backtest", {"ticker": "AAPL", "strategies": ["momentum", "mean_reversion", "rsi", "macd"]}, "多策略回测"),
        
        # 风险指标
        ("POST", "/api/risk_metrics", {"transactions": [{"day": 1, "operate": "buy", "price": 100, "total_balance": 10000}]}, "风险指标计算"),
        
        # 仓位计算
        ("POST", "/api/position/calculate", {"method": "fixed_fractional", "total_capital": 10000, "risk_per_trade": 0.02, "ticker": "AAPL"}, "仓位计算"),
        
        # 止损设置
        ("POST", "/api/stop_loss/set", {"ticker": "AAPL", "price": 150, "trailing_percent": 5}, "止损设置"),
        ("GET", "/api/stop_loss/check/AAPL/140", None, "止损检查"),
        
        # 市场搜索
        ("GET", "/api/market/search", {"q": "AAPL", "limit": 5}, "股票搜索"),
        ("GET", "/api/market/a_stocks", {"limit": 10}, "A股列表"),
        
        # 报告导出
        ("POST", "/api/report/export", {"ticker": "AAPL", "format": "pdf"}, "报告导出"),
    ]
    
    for method, endpoint, data, desc in advanced_tests:
        total_tests += 1
        if test_endpoint(method, endpoint, data, desc):
            passed_tests += 1
        time.sleep(0.5)
    
    # 3. 用户管理功能测试
    print("\n" + "="*80)
    print("用户管理功能测试")
    print("="*80)
    
    user_tests = [
        # 用户注册（简化版，无需验证码）
        ("POST", "/api/register", {"username": "testuser", "password": "testpass", "password_confirm": "testpass", "email": "test@example.com"}, "用户注册"),
        
        # 用户登录
        ("POST", "/api/login", {"username": "testuser", "password": "testpass"}, "用户登录"),
        
        # 密码找回申请
        ("POST", "/api/forgot_password", {"username": "testuser"}, "密码找回申请"),
        
        # 用户资料
        ("GET", "/api/user/profile", None, "用户资料"),
        
        # 退出登录
        ("POST", "/api/logout", None, "退出登录"),
    ]
    
    for method, endpoint, data, desc in user_tests:
        total_tests += 1
        if test_endpoint(method, endpoint, data, desc):
            passed_tests += 1
        time.sleep(0.5)
    
    # 4. 专业功能测试
    print("\n" + "="*80)
    print("专业功能测试")
    print("="*80)
    
    professional_tests = [
        # 投资组合优化
        ("POST", "/api/advanced/portfolio/optimize", {"tickers": ["AAPL", "MSFT", "GOOGL"], "risk_tolerance": 0.1, "target_return": 0.15}, "投资组合优化"),
        
        # 多时间框架分析
        ("POST", "/api/advanced/timeframe/analyze", {"ticker": "AAPL", "timeframes": ["1D", "1W", "1M"]}, "多时间框架分析"),
        
        # 机器学习流水线
        ("POST", "/api/advanced/ml/pipeline", {"ticker": "AAPL", "models": ["RandomForest", "XGBoost"]}, "机器学习流水线"),
        
        # 风险管理
        ("POST", "/api/advanced/risk/institutional", {"portfolio": {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.4}, "confidence_level": 0.95}, "机构风险管理"),
        
        # 合规检查
        ("POST", "/api/advanced/compliance/check", {"portfolio": {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.4}, "rules": ["concentration_limit", "var_limit"]}, "合规检查"),
    ]
    
    for method, endpoint, data, desc in professional_tests:
        total_tests += 1
        if test_endpoint(method, endpoint, data, desc):
            passed_tests += 1
        time.sleep(0.5)
    
    # 5. 调试功能测试
    print("\n" + "="*80)
    print("调试功能测试")
    print("="*80)
    
    debug_tests = [
        ("GET", "/__routes", None, "路由列表"),
        ("GET", "/__paths", None, "路径信息"),
    ]
    
    for method, endpoint, data, desc in debug_tests:
        total_tests += 1
        if test_endpoint(method, endpoint, data, desc):
            passed_tests += 1
        time.sleep(0.5)
    
    # 测试结果汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("系统功能良好，大部分功能正常工作！")
    elif success_rate >= 60:
        print("系统功能基本正常，但部分功能需要修复")
    else:
        print("系统功能存在严重问题，需要紧急修复")
    
    print(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
