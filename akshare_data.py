#!/usr/bin/env python3
"""
A股数据获取模块 - 基于AKShare
支持沪深北三市、指数、ETF等全市场数据
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time
from typing import Optional, Dict, Any


def _fetch_a_share_with_proxy_fallback(symbol: str, start_ak: str, end_ak: str, adjust: str):
    """获取A股数据，代理失败时自动禁用代理重试"""
    try:
        return ak.stock_zh_a_hist(symbol=symbol, start_date=start_ak, end_date=end_ak, adjust=adjust)
    except Exception as e:
        err_str = str(e).lower()
        if 'proxy' in err_str or 'proxierror' in err_str or 'remotedisconnected' in err_str:
            print("检测到代理错误，尝试禁用代理直连...")
            saved = {k: os.environ.pop(k, None) for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy')}
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_ak, end_date=end_ak, adjust=adjust)
                print("直连成功")
                return df
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        raise


def download_a_share(stock_code: str,
                    start_date: str,
                    end_date: str,
                    out_dir: str = "data",
                    adjust: str = "qfq") -> str:
    """
    下载A股日线数据（前复权）并保存为CSV
    
    Args:
        stock_code: 股票代码，如 600519, 000001, 430047
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD  
        out_dir: 输出目录
        adjust: 复权类型 qfq(前复权)/hfq(后复权)/None(不复权)
    
    Returns:
        str: CSV文件路径
        
    Raises:
        ValueError: 数据获取失败
    """
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{stock_code}.csv")
    
    try:
        # 统一按北京时间将截止日期补齐到今日
        try:
            beijing_today = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
        except Exception:
            beijing_today = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d')
        if (not end_date) or (end_date < beijing_today):
            end_date = beijing_today

        # AKShare日期格式转换
        start_ak = start_date.replace("-", "")
        end_ak = end_date.replace("-", "")
        
        print(f"正在获取A股数据: {stock_code} ({start_date} 到 {end_date})")
        
        # 获取股票历史数据（代理异常时自动禁用代理重试）
        df = _fetch_a_share_with_proxy_fallback(stock_code, start_ak, end_ak, adjust)
        
        if df.empty:
            raise ValueError(f"AKShare返回空数据，请检查股票代码 {stock_code} 或日期范围")
        
        # 统一列名格式，与Yahoo Finance保持一致
        df.rename(columns={
            "日期": "Date",
            "开盘": "Open", 
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
            "成交额": "Amount",
            "振幅": "Amplitude",
            "涨跌幅": "Change",
            "涨跌额": "ChangeAmount",
            "换手率": "Turnover"
        }, inplace=True)
        
        # 确保Date列为datetime类型
        df["Date"] = pd.to_datetime(df["Date"])
        
        # 按日期排序
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 保存核心OHLCV字段（与美股格式完全一致）
        core_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_core = df[core_columns].copy()
        
        # 数据质量检查
        if len(df_core) < 10:
            raise ValueError(f"数据量不足，仅 {len(df_core)} 行，请检查日期范围")
        
        # 检查是否有缺失值
        missing_count = df_core.isnull().sum().sum()
        if missing_count > 0:
            print(f"警告: 发现 {missing_count} 个缺失值，将进行前向填充")
            df_core = df_core.fillna(method='ffill')
        
        # 保存到CSV
        df_core.to_csv(file_path, index=False)
        
        print(f"[OK] A股数据下载成功: {file_path}")
        print(f"   数据量: {len(df_core)} 行")
        print(f"   日期范围: {df_core['Date'].min().strftime('%Y-%m-%d')} 到 {df_core['Date'].max().strftime('%Y-%m-%d')}")
        
        return file_path
        
    except Exception as e:
        error_msg = f"获取A股数据失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

def get_stock_list() -> pd.DataFrame:
    """
    获取A股股票列表
    
    Returns:
        pd.DataFrame: 包含股票代码、名称、市场等信息的DataFrame
    """
    try:
        # 获取沪深A股列表
        stock_list = ak.stock_zh_a_spot_em()
        return stock_list[['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额']].head(100)
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()

def get_market_index() -> pd.DataFrame:
    """
    获取主要市场指数
    
    Returns:
        pd.DataFrame: 指数数据
    """
    try:
        # 获取主要指数
        indices = ['sh000001', 'sz399001', 'sz399006']  # 上证、深证、创业板
        index_data = []
        
        for idx in indices:
            try:
                df = ak.stock_zh_index_daily(symbol=idx)
                if not df.empty:
                    latest = df.iloc[-1]
                    index_data.append({
                        '指数': idx,
                        '最新价': latest['收盘'],
                        '涨跌幅': latest['涨跌幅'],
                        '成交量': latest['成交量']
                    })
            except:
                continue
                
        return pd.DataFrame(index_data)
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return pd.DataFrame()

def download_us_stock(ticker: str, start_date: str, end_date: str, out_dir: str = "data") -> str:
    """
    下载美股日线数据（yfinance）并保存为CSV，格式与A股一致
    
    Args:
        ticker: 股票代码，如 AAPL, MSFT
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        out_dir: 输出目录
    Returns:
        str: CSV文件路径
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ValueError("请安装 yfinance: pip install yfinance")
    
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{ticker}.csv")
    
    def _fetch():
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty or len(df) < 10:
            raise ValueError(f"yfinance 返回空或数据不足，请检查 {ticker} 或日期范围")
        df = df.reset_index()
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"缺少列 {c}")
        df = df[cols].copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(file_path, index=False)
        return file_path
    
    try:
        return _fetch()
    except Exception as e:
        err_str = str(e).lower()
        if 'proxy' in err_str or 'connect' in err_str or 'timeout' in err_str:
            print("检测到网络/代理问题，尝试禁用代理直连...")
            saved = {k: os.environ.pop(k, None) for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy')}
            try:
                fp = _fetch()
                print("直连成功")
                return fp
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        raise


def validate_stock_code(stock_code: str) -> bool:
    """
    验证A股股票代码格式
    
    Args:
        stock_code: 股票代码
        
    Returns:
        bool: 是否为有效的A股代码
    """
    if not stock_code or len(stock_code) < 6:
        return False
    
    # 沪市主板/科创板 600xxx, 601xxx, 603xxx, 605xxx, 688xxx
    if stock_code.startswith(('600', '601', '603', '605', '688')):
        return True
    
    # 深市主板 000xxx, 001xxx
    if stock_code.startswith(('000', '001')):
        return True
        
    # 深市中小板 002xxx
    if stock_code.startswith('002'):
        return True
        
    # 深市创业板 300xxx
    if stock_code.startswith('300'):
        return True
        
    # 北交所 430xxx, 830xxx, 870xxx
    if stock_code.startswith(('430', '830', '870')):
        return True
        
    return False

if __name__ == "__main__":
    # 测试A股数据下载
    print("=== A股数据获取测试 ===")
    
    # 测试股票列表
    print("\n1. 获取股票列表...")
    stock_list = get_stock_list()
    if not stock_list.empty:
        print("热门A股:")
        print(stock_list.head(10))
    
    # 测试指数
    print("\n2. 获取市场指数...")
    index_data = get_market_index()
    if not index_data.empty:
        print("主要指数:")
        print(index_data)
    
    # 测试个股数据下载
    print("\n3. 测试个股数据下载...")
    test_codes = ['600519', '000001', '300750', '603175', '688498', '603662']  # 含用户关注标的
    
    for code in test_codes:
        try:
            if validate_stock_code(code):
                print(f"\n正在测试 {code}...")
                file_path = download_a_share(
                    stock_code=code,
                    start_date='2024-01-01',
                    end_date='2024-12-31'
                )
                print(f"[OK] {code} 数据已保存到: {file_path}")
            else:
                print(f"[ERROR] {code} 不是有效的A股代码")
        except Exception as e:
            print(f"[ERROR] {code} 下载失败: {e}")
        
        time.sleep(1)  # 避免请求过快
    
    print("\n=== 测试完成 ===")
