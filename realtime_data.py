#!/usr/bin/env python3
"""
实时数据获取与更新模块
支持Yahoo Finance API、Alpha Vantage等数据源
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import asyncio
import aiohttp
from typing import Dict, List, Optional, Callable, Any
import json
import os
from enhanced_database import init_enhanced_database_system
import warnings
warnings.filterwarnings('ignore')

# 条件导入可选依赖
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    print("警告: Schedule未安装，定时任务功能将不可用")

class RealTimeDataProvider:
    """实时数据提供者基类"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = {}
        self.last_update = {}
    
    async def get_realtime_price(self, ticker: str) -> Dict[str, Any]:
        """获取实时价格"""
        raise NotImplementedError
    
    async def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """获取历史数据"""
        raise NotImplementedError

class YahooFinanceProvider(RealTimeDataProvider):
    """Yahoo Finance数据提供者"""
    
    def __init__(self):
        super().__init__()
        self.session = None
    
    async def get_realtime_price(self, ticker: str) -> Dict[str, Any]:
        """获取Yahoo Finance实时价格"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 获取最新价格数据
            hist = stock.history(period="1d", interval="1m")
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            return {
                'ticker': ticker,
                'price': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'timestamp': datetime.now(),
                'change': latest['Close'] - latest['Open'],
                'change_percent': (latest['Close'] - latest['Open']) / latest['Open'] * 100,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield')
            }
        except Exception as e:
            print(f"获取{ticker}实时数据失败: {e}")
            return None
    
    async def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """获取历史数据"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return pd.DataFrame()
            
            # 重置索引，使Date成为列
            data.reset_index(inplace=True)
            
            # 添加技术指标
            data = self.calculate_technical_indicators(data)
            
            return data
        except Exception as e:
            print(f"获取{ticker}历史数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if len(data) < 20:
            return data
        
        # 移动平均线
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        # 布林带
        data['SMA'] = data['Close'].rolling(window=20).mean()
        data['Std_dev'] = data['Close'].rolling(window=20).std()
        data['Upper_band'] = data['SMA'] + (data['Std_dev'] * 2)
        data['Lower_band'] = data['SMA'] - (data['Std_dev'] * 2)
        
        # VWAP
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        return data

class RealTimeDataManager:
    """实时数据管理器"""
    
    def __init__(self, db = None):
        self.providers = {}
        self.subscribers = {}  # 订阅者回调函数
        self.running = False
        self.update_interval = 60  # 更新间隔（秒）
        self.db = db
        
        # 添加默认提供者
        self.add_provider('yahoo', YahooFinanceProvider())
    
    def add_provider(self, name: str, provider: RealTimeDataProvider):
        """添加数据提供者"""
        self.providers[name] = provider
    
    def subscribe(self, ticker: str, callback: Callable[[Dict], None]):
        """订阅股票数据更新"""
        if ticker not in self.subscribers:
            self.subscribers[ticker] = []
        self.subscribers[ticker].append(callback)
    
    def unsubscribe(self, ticker: str, callback: Callable[[Dict], None]):
        """取消订阅"""
        if ticker in self.subscribers:
            self.subscribers[ticker].remove(callback)
            if not self.subscribers[ticker]:
                del self.subscribers[ticker]
    
    async def get_realtime_data(self, ticker: str, provider: str = 'yahoo') -> Dict[str, Any]:
        """获取实时数据"""
        if provider not in self.providers:
            raise ValueError(f"未知的数据提供者: {provider}")
        
        return await self.providers[provider].get_realtime_price(ticker)
    
    async def update_market_data_cache(self, ticker: str):
        """更新市场数据缓存"""
        try:
            data = await self.get_realtime_data(ticker)
            if data:
                # 存储到数据库缓存
                conn = self.db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data_cache 
                    (ticker, date, open_price, high_price, low_price, close_price, volume, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    datetime.now().date(),
                    data['open'],
                    data['high'],
                    data['low'],
                    data['price'],
                    data['volume'],
                    datetime.now()
                ))
                
                conn.commit()
                conn.close()
                
                # 通知订阅者
                if ticker in self.subscribers:
                    for callback in self.subscribers[ticker]:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"回调函数执行失败: {e}")
        
        except Exception as e:
            print(f"更新{ticker}数据缓存失败: {e}")
    
    def start_realtime_updates(self, tickers: List[str]):
        """启动实时数据更新"""
        self.running = True
        
        async def update_loop():
            while self.running:
                tasks = []
                for ticker in tickers:
                    tasks.append(self.update_market_data_cache(ticker))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(self.update_interval)
        
        # 在新线程中运行异步循环
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(update_loop())
        
        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()
        
        print(f"已启动实时数据更新，监控股票: {tickers}")
    
    def stop_realtime_updates(self):
        """停止实时数据更新"""
        self.running = False
        print("已停止实时数据更新")

class DataUpdater:
    """数据更新器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.provider = YahooFinanceProvider()
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    async def update_single_stock(self, ticker: str, period: str = "max") -> bool:
        """更新单个股票数据"""
        try:
            print(f"更新 {ticker} 数据...")
            
            # 获取新数据
            new_data = await self.provider.get_historical_data(ticker, period)
            
            if new_data.empty:
                print(f"无法获取 {ticker} 的数据")
                return False
            
            # 检查是否存在本地文件
            file_path = os.path.join(self.data_dir, f"{ticker}.csv")
            
            if os.path.exists(file_path):
                # 读取现有数据
                existing_data = pd.read_csv(file_path)
                existing_data['Date'] = pd.to_datetime(existing_data['Date'])
                
                # 找到最后一个日期
                last_date = existing_data['Date'].max()
                
                # 只保留新数据
                new_data['Date'] = pd.to_datetime(new_data['Date'])
                incremental_data = new_data[new_data['Date'] > last_date]
                
                if not incremental_data.empty:
                    # 合并数据
                    combined_data = pd.concat([existing_data, incremental_data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                    combined_data = combined_data.sort_values('Date')
                    
                    # 重新计算技术指标
                    combined_data = self.provider.calculate_technical_indicators(combined_data)
                    
                    # 保存更新后的数据
                    combined_data.to_csv(file_path, index=False)
                    print(f"✅ {ticker} 数据已更新 ({len(incremental_data)} 新记录)")
                else:
                    print(f"📊 {ticker} 数据已是最新")
            else:
                # 保存新文件
                new_data.to_csv(file_path, index=False)
                print(f"✅ {ticker} 数据已创建 ({len(new_data)} 记录)")
            
            return True
            
        except Exception as e:
            print(f"❌ 更新 {ticker} 失败: {e}")
            return False
    
    async def update_all_stocks(self, tickers: List[str]) -> Dict[str, bool]:
        """批量更新股票数据"""
        results = {}
        
        print(f"开始更新 {len(tickers)} 支股票的数据...")
        
        # 并发更新（限制并发数量避免被限流）
        semaphore = asyncio.Semaphore(5)  # 最多5个并发请求
        
        async def update_with_semaphore(ticker):
            async with semaphore:
                result = await self.update_single_stock(ticker)
                await asyncio.sleep(1)  # 避免请求过快
                return ticker, result
        
        tasks = [update_with_semaphore(ticker) for ticker in tickers]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results_list:
            if isinstance(result, tuple):
                ticker, success = result
                results[ticker] = success
            else:
                print(f"更新过程中出现异常: {result}")
        
        successful = sum(1 for success in results.values() if success)
        print(f"\n数据更新完成: {successful}/{len(tickers)} 成功")
        
        return results
    
    def schedule_daily_update(self, tickers: List[str], update_time: str = "09:00"):
        """计划每日数据更新"""
        if not HAS_SCHEDULE:
            print("Schedule未安装，无法设置定时任务")
            return
        
        def run_update():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.update_all_stocks(tickers))
        
        schedule.every().day.at(update_time).do(run_update)
        
        print(f"已设置每日 {update_time} 自动更新数据")
        
        # 运行调度器
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

class MarketDataAPI:
    """市场数据API接口"""
    
    def __init__(self, data_manager: RealTimeDataManager):
        self.data_manager = data_manager
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        major_indices = ['SPY', 'QQQ', 'DIA', 'IWM']  # 主要指数ETF
        
        overview = {}
        for ticker in major_indices:
            data = await self.data_manager.get_realtime_data(ticker)
            if data:
                overview[ticker] = {
                    'price': data['price'],
                    'change': data['change'],
                    'change_percent': data['change_percent']
                }
        
        return overview
    
    async def get_sector_performance(self) -> Dict[str, Any]:
        """获取行业表现"""
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication': 'XLC'
        }
        
        performance = {}
        for sector, ticker in sector_etfs.items():
            data = await self.data_manager.get_realtime_data(ticker)
            if data:
                performance[sector] = {
                    'ticker': ticker,
                    'price': data['price'],
                    'change_percent': data['change_percent']
                }
        
        return performance

# 使用示例和测试
async def main():
    """主函数示例"""
    
    # 创建数据管理器
    data_manager = RealTimeDataManager()
    
    # 测试实时数据获取
    print("=== 测试实时数据获取 ===")
    aapl_data = await data_manager.get_realtime_data('AAPL')
    if aapl_data:
        print(f"AAPL 当前价格: ${aapl_data['price']:.2f}")
        print(f"涨跌幅: {aapl_data['change_percent']:.2f}%")
    
    # 测试数据更新
    print("\n=== 测试数据更新 ===")
    updater = DataUpdater()
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    results = await updater.update_all_stocks(test_tickers)
    print("更新结果:", results)
    
    # 测试市场概览
    print("\n=== 测试市场概览 ===")
    api = MarketDataAPI(data_manager)
    market_overview = await api.get_market_overview()
    
    for ticker, data in market_overview.items():
        print(f"{ticker}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
