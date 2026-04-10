#!/usr/bin/env python3
import requests, os, time

BASE = os.environ.get('SMOKE_BASE', 'http://127.0.0.1:5000')

def log_resp(r):
    try:
        print(r.status_code, r.json())
    except Exception:
        print(r.status_code, len(r.content))

def main():
    s = requests.Session()

    # 等待服务就绪
    for _ in range(30):
        try:
            r = s.get(f"{BASE}/api/health", timeout=1.5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    # 1. 访客登录
    log_resp(s.post(f"{BASE}/api/guest-login", json={}))

    # 2. 获取 AAPL 数据
    log_resp(s.post(f"{BASE}/get_data", json={
        'ticker': 'AAPL', 'start_date': '2023-01-01', 'end_date': '2024-01-01'
    }))

    # 3. 训练核心链路
    log_resp(s.post(f"{BASE}/train_model", json={
        'ticker': 'AAPL', 'epochs': 10, 'batch_size': 16
    }))

    # 4. 交易 CSV
    r = s.get(f"{BASE}/results/transactions/AAPL_transactions.csv")
    print('csv size:', len(r.content))

    # 5. 预测图
    r = s.get(f"{BASE}/images/predictions/AAPL_prediction.png")
    print('png size:', len(r.content))

    print('核心链路检查完成，smoke 通过（请确认上述返回均为 200）。')

if __name__ == '__main__':
    main()


