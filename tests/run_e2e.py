#!/usr/bin/env python3
import json
import sys


def main():
    from enhanced_flask_interface_advanced import app

    ok = 0
    fail = 0

    def expect(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"[FAIL] {msg}")

    with app.test_client() as c:
        # health
        r = c.get('/api/health')
        expect(r.status_code == 200 and r.json.get('success') is True, 'health')

        # guest login
        r = c.post('/api/guest-login')
        expect(r.status_code == 200 and r.json.get('success') is True and r.json.get('guest') is True, 'guest-login')

        # register (simplified, no verification code)
        r = c.post('/api/register', json={'username': 'user_demo', 'password': 'p@ss12345', 'password_confirm': 'p@ss12345', 'email': 'demo@example.com'})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'register reachable')

        # forgot password
        r = c.post('/api/forgot_password', json={'username': 'user_demo'})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'forgot_password reachable')

        # backtest (guest allowed)
        r = c.post('/api/strategies/backtest', json={'ticker': 'AAPL'})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'backtest reachable')

        # position sizing (fixed_fractional)
        r = c.post('/api/position/calculate', json={'total_capital': 10000, 'method': 'fixed_fractional', 'risk_per_trade': 0.02, 'ticker': 'AAPL'})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'position calculate reachable')

        # risk metrics with synthetic transactions
        tx = [
            {'Date': '2024-01-01', 'operate': 'buy', 'price': 100, 'total_balance': 10000},
            {'Date': '2024-01-02', 'operate': 'sell', 'price': 105, 'total_balance': 10200},
            {'Date': '2024-01-03', 'operate': 'buy', 'price': 103, 'total_balance': 10150},
            {'Date': '2024-01-04', 'operate': 'sell', 'price': 110, 'total_balance': 10300},
        ]
        r = c.post('/api/risk_metrics', json={'transactions': tx})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'risk metrics reachable')

        # get_data to materialize tmp csv (requires local data/AAPL.csv present in repo)
        r = c.post('/get_data', json={'ticker': 'AAPL', 'start_date': '2020-01-01', 'end_date': '2020-12-31'})
        expect(r.status_code == 200 and r.json.get('success') in (True, False), 'get_data reachable')

        # file serving endpoints: expect 200 or 404 depending on existence
        r = c.get('/tmp/flask/ticker/AAPL.csv')
        expect(r.status_code in (200, 404), 'serve tmp ticker csv returns 200/404')
        r = c.get('/results/transactions/AAPL_transactions.csv')
        expect(r.status_code in (200, 404), 'serve transactions csv returns 200/404')

    print(f"E2E checks passed: {ok}, failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())


