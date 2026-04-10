#!/usr/bin/env python3
"""
增强 Flask 应用 - 集成所有功能
Enhanced Flask Application with Advanced Features
"""
import os, sys, json, pandas as pd, numpy as np, torch
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, request, jsonify, session, send_file, send_from_directory,
                   render_template, redirect, url_for)
import jwt, hashlib, warnings, asyncio, time, logging, re
from werkzeug.utils import secure_filename
warnings.filterwarnings('ignore')

# 性能分析工具
def ts(): return time.time()

# ---------- 内部工具 ----------
from enhanced_database import init_enhanced_database_system
from stock_prediction_lstm import predict, format_feature
from RLagent import process_stock
from risk_management import RiskManager, StopLossManager, PositionSizer
from trading_strategies import (
    TradingSystem,
    MomentumStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    MACDStrategy,
    MovingAverageCrossoverStrategy,
    BreakoutStrategy,
)
from realtime_data import RealTimeDataManager, DataUpdater, MarketDataAPI
from pdf_report_generator import generate_analysis_report
from datetime import datetime
from akshare_data import download_a_share, download_us_stock, validate_stock_code, get_stock_list
from a_share_feature_engineering import process_a_share_data

# ---------- 高级功能模块 ----------
from portfolio_optimizer import PortfolioOptimizer
from multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ml_pipeline import MLPipeline
from api_ecosystem import APIEcosystem
from institutional_risk_management import InstitutionalRiskManager
from multi_account_manager import MultiAccountManager
from compliance_tools import ComplianceManager
from white_label_solution import WhiteLabelManager
from permissions import PermissionManager, require_admin

# ---------- 基础配置 ----------
app = Flask(__name__)
app.secret_key = 'enhanced-trading-system-secret-key-2024'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
# 统一使用绝对路径，避免工作目录差异导致的404
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'results')
TEMP_DIR  = os.path.join(BASE_DIR, 'tmp', 'flask')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
AVATAR_UPLOAD_DIR = os.path.join(UPLOADS_DIR, 'avatars')
ALLOWED_AVATAR_EXTS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2MB
for d in [TEMP_DIR, UPLOADS_DIR, AVATAR_UPLOAD_DIR, os.path.join(SAVE_DIR, 'pic', 'predictions'), os.path.join(SAVE_DIR, 'pic', 'loss'), os.path.join(SAVE_DIR, 'pic', 'earnings'), os.path.join(SAVE_DIR, 'pic', 'trades'), os.path.join(SAVE_DIR, 'transactions')]:
    os.makedirs(d, exist_ok=True)

# ---------- 全局实例 ----------
db = init_enhanced_database_system()
PermissionManager._db = db
risk_m    = RiskManager()
stop_m    = StopLossManager()
data_mgr  = RealTimeDataManager()
market_api= MarketDataAPI(data_mgr)

# ---------- 高级功能实例 ----------
portfolio_optimizer = PortfolioOptimizer()
timeframe_analyzer = MultiTimeframeAnalyzer()
ml_pipeline = MLPipeline()
api_ecosystem = APIEcosystem(app)
institutional_risk = InstitutionalRiskManager()
multi_account = MultiAccountManager()
compliance = ComplianceManager()
white_label = WhiteLabelManager()

# 缓存控制中间件（此时 app 已定义）
@app.after_request
def no_cache(resp):
    try:
        if request.endpoint in ['train_model', 'get_data', 'run_analysis']:
            resp.headers['Cache-Control'] = 'no-store, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp

# ---------- 工具函数 ----------
def hash_pwd(p):
    """兼容旧代码的快捷哈希（实际使用 bcrypt）"""
    return db['user_manager'].hash_password(p)

def verify_pwd(p, h):
    """统一使用 UserManager 的验证"""
    return db['user_manager'].verify_password(p, h)

def gen_token(uid):
    return jwt.encode({'user_id': uid, 'exp': datetime.utcnow() + timedelta(hours=24)},
                      app.config['JWT_SECRET_KEY'], algorithm='HS256')

def login_required(f):
    @wraps(f)
    def wrap(*a, **k):
        user_id = session.get('user_id')
        is_guest = session.get('guest') is True
        
        # 检查 JWT（API 调用）
        if not user_id and not is_guest:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                try:
                    token = auth_header.split(' ')[1]
                    payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
                    user_id = payload.get('user_id')
                except jwt.ExpiredSignatureError:
                    return jsonify(success=False, message='登录已过期'), 401
                except jwt.InvalidTokenError:
                    return jsonify(success=False, message='无效的令牌'), 401
        
        if not user_id and not is_guest:
            return jsonify(success=False, message='请先登录'), 401
        
        request.current_user_id = user_id
        request.is_guest = is_guest
        return f(*a, **k)
    return wrap

# ---------- 基础路由 ----------
@app.route('/')
def index():
    return render_template('index_advanced.html')


@app.route('/profile')
def profile_page():
    # 个人中心仅允许已登录用户访问（游客/未登录都跳登录页）
    if not session.get('user_id'):
        return redirect(url_for('login_test'))
    return render_template('profile.html')

@app.route('/login-test')
@app.route('/login')
def login_test():
    return render_template('login_test.html')

@app.route('/admin')
def admin_page():
    """管理员后台页面，仅管理员可访问"""
    if session.get('role') != 'admin':
        return redirect(url_for('login_test'))
    return render_template('admin_dashboard.html')

@app.route('/api/health')
def health():
    return jsonify(status='healthy', success=True, ts=datetime.now())

# 调试：列出已注册路由，排查 404 原因】是否跑的就是本文件、路由是否存在【
@app.route('/__routes')
def list_routes():
    try:
        rules = []
        for r in app.url_map.iter_rules():
            rules.append({
                'rule': str(r),
                'methods': sorted([m for m in r.methods if m not in ('HEAD','OPTIONS')])
            })
        return jsonify(success=True, routes=sorted(rules, key=lambda x: x['rule']))
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/__paths')
def show_paths():
    return jsonify(success=True, base_dir=BASE_DIR, save_dir=SAVE_DIR, temp_dir=TEMP_DIR)

# ========== 用户认证（简化版，无邮箱验证）==========

@app.route('/api/register', methods=['POST'])
def register():
    """
    简化注册流程 - 无需邮箱验证
    1. 填写用户名、密码、确认密码
    2. 可选填写邮箱（用于密码找回，但不验证）
    3. 立即完成注册，可选自动登录
    """
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    password_confirm = (data.get('password_confirm') or '').strip()
    email = (data.get('email') or '').strip()
    auto_login = data.get('auto_login', True)
    
    if not username:
        return jsonify(success=False, message='请输入用户名')
    if not password:
        return jsonify(success=False, message='请输入密码')
    if password != password_confirm:
        return jsonify(success=False, message='两次输入的密码不一致')
    if len(password) < 6:
        return jsonify(success=False, message='密码长度至少6位')
    if len(password) > 20:
        return jsonify(success=False, message='密码长度不能超过20位')
    
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return jsonify(success=False, message='用户名只能包含3-20位字母、数字或下划线')
    
    if db['user_manager'].get_user_by_username(username):
        return jsonify(success=False, message='用户名已存在')
    
    final_email = None
    if email:
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return jsonify(success=False, message='邮箱格式不正确')
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify(success=False, message='该邮箱已被其他用户使用')
        conn.close()
        final_email = email
    else:
        final_email = f"{username}@placeholder.local"
    
    result = db['user_manager'].register_user(username, password, final_email)
    
    if not result.get('success'):
        return jsonify(success=False, message=result.get('message', '注册失败'))

    user_id = result.get('user_id')
    role = result.get('role', 'user')
    
    try:
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (?, 'REGISTER', ?, ?, ?)
        ''', (user_id, f'User registered: {username}', request.remote_addr, datetime.now()))
        conn.commit()
        conn.close()
    except Exception:
        pass
    
    if auto_login:
        session['user_id'] = user_id
        session['username'] = username
        session['role'] = role
        session.pop('guest', None)
        db['user_manager'].update_last_login(user_id)
        token = gen_token(user_id)
        return jsonify(
            success=True, message='注册成功，已自动登录',
            user_id=user_id, user={'id': user_id, 'username': username, 'email': final_email, 'role': role, 'loggedIn': True},
            token=token, auto_login=True
        )
    else:
        return jsonify(success=True, message='注册成功，请登录', user_id=user_id, auto_login=False)


@app.route('/api/login', methods=['POST'])
def login():
    """修复版登录 - 失败不自动转游客，增加安全日志"""
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password')
    
    if not username or not password:
        return jsonify(success=False, message='请提供用户名和密码')
    
    u = db['user_manager'].get_user_by_username(username)
    
    if not u or not db['user_manager'].verify_password(password, u['password_hash']):
        try:
            conn = db['db'].get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
                VALUES (?, 'LOGIN_FAILED', ?, ?, ?)
            ''', (u['id'] if u else None, f'Failed login for: {username}', request.remote_addr, datetime.now()))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return jsonify(success=False, message='用户名或密码错误')
    
    if u.get('is_active', 1) != 1:
        return jsonify(success=False, message='账户已被禁用，请联系管理员')
    
    if not u['password_hash'].startswith('$2'):
        try:
            new_hash = db['user_manager'].hash_password(password)
            conn = db['db'].get_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, u['id']))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"密码迁移失败: {e}")
    
    db['user_manager'].update_last_login(u['id'])
    session['user_id'] = u['id']
    session['username'] = u['username']
    session['role'] = u.get('role', 'user')
    session.pop('guest', None)
    token = gen_token(u['id'])
    
    try:
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (?, 'LOGIN_SUCCESS', ?, ?, ?)
        ''', (u['id'], f'Login from IP: {request.remote_addr}', request.remote_addr, datetime.now()))
        conn.commit()
        conn.close()
    except Exception:
        pass
    
    return jsonify(
        success=True, message='登录成功',
        user={'id': u['id'], 'username': u['username'], 'email': u.get('email'), 'role': u.get('role', 'user'), 'loggedIn': True},
        token=token, guest=False
    )


@app.route('/api/guest-login', methods=['POST'])
def guest_login():
    """独立的游客登录入口 - 不再从失败登录跳转"""
    session.clear()
    session['guest'] = True
    session['guest_id'] = f"guest_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()[:8]}"
    try:
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (NULL, 'GUEST_LOGIN', ?, ?, ?)
        ''', (f'Guest login from IP: {request.remote_addr}', request.remote_addr, datetime.now()))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return jsonify(
        success=True, message='游客登录成功（部分功能受限）',
                       user={'id': None, 'username': 'guest', 'role': 'guest', 'loggedIn': True},
        guest=True
    )


@app.route('/api/logout', methods=['POST'])
def logout():
    user_id = session.get('user_id')
    session.clear()
    if user_id:
        try:
            conn = db['db'].get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES (?, 'LOGOUT', ?, ?)
            ''', (user_id, 'User logout', datetime.now()))
            conn.commit()
            conn.close()
        except Exception:
            pass
    return jsonify(success=True, message='已退出登录')


@app.route('/api/forgot_password', methods=['POST'])
def forgot_password():
    """
    密码找回：
    - 用户绑定真实邮箱：重置码通过邮件发送给用户
    - 用户未绑定邮箱：重置码由管理员在后台查看并告知
    """
    from email_verification import mail_service

    data = request.get_json()
    username = (data.get('username') or '').strip()
    if not username:
        return jsonify(success=False, message='请提供用户名')

    u = db['user_manager'].get_user_by_username(username)
    if not u:
        return jsonify(success=True, message='如果该用户名存在，重置码已生成并发送')

    reset_code = f"RESET-{u['id']}-{os.urandom(4).hex()[:8].upper()}"

    try:
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE password_resets SET is_used = 1 WHERE user_id = ?', (u['id'],))
        expires_at = datetime.now() + timedelta(minutes=10)
        cursor.execute('''
            INSERT INTO password_resets (user_id, reset_code, expires_at)
            VALUES (?, ?, ?)
        ''', (u['id'], reset_code, expires_at))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify(success=False, message=f'生成重置码失败: {e}')

    user_email = u.get('email', '')
    has_real_email = bool(user_email) and '@placeholder.local' not in user_email
    email_sent = False
    if has_real_email:
        email_sent = mail_service.send_reset_code_email(user_email, username, reset_code)

    print(f"\n{'='*60}")
    print(f"[密码重置请求] 用户名: {username}")
    print(f"重置码: {reset_code}")
    print(f"邮件发送: {'成功 → ' + user_email if email_sent else '未发送（无邮箱或发送失败）'}")
    print(f"有效期: 10分钟")
    print(f"{'='*60}\n")

    if email_sent:
        masked = user_email[:3] + '***' + user_email[user_email.index('@'):]
        return jsonify(
            success=True,
            message=f'重置码已发送到邮箱 {masked}，请查收（10分钟内有效）',
            email_sent=True
        )
    else:
        return jsonify(
            success=True,
            message='重置码已生成，请联系管理员获取（10分钟内有效）',
            email_sent=False,
            hint='管理员可在后台“密码重置记录”中查看重置码'
        )


@app.route('/api/reset_password', methods=['POST'])
def reset_password():
    """使用重置码修改密码"""
    data = request.get_json()
    username = (data.get('username') or '').strip()
    reset_code = (data.get('reset_code') or '').strip()
    new_password = data.get('new_password')
    
    if not all([username, reset_code, new_password]):
        return jsonify(success=False, message='缺少必要参数')
    if len(new_password) < 6:
        return jsonify(success=False, message='密码至少6位')
    
    u = db['user_manager'].get_user_by_username(username)
    if not u:
        return jsonify(success=False, message='无效的用户名或重置码')
    
    try:
        conn = db['db'].get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, expires_at, is_used FROM password_resets
            WHERE user_id = ? AND reset_code = ? AND is_used = 0
            ORDER BY created_at DESC LIMIT 1
        ''', (u['id'], reset_code))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return jsonify(success=False, message='无效的重置码')
        
        rid, expires_at_str, is_used = result
        try:
            expires_at_str = str(expires_at_str)
            expires_at = datetime.fromisoformat(expires_at_str.replace(' ', 'T', 1)) if ' ' in expires_at_str else datetime.fromisoformat(expires_at_str)
        except (ValueError, TypeError):
            expires_at = datetime.now() - timedelta(days=1)
        if datetime.now() > expires_at:
            conn.close()
            return jsonify(success=False, message='重置码已过期')
        
        new_hash = db['user_manager'].hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, u['id']))
        cursor.execute('UPDATE password_resets SET is_used = 1 WHERE id = ?', (rid,))
        conn.commit()
        conn.close()
        
        try:
            conn2 = db['db'].get_connection()
            cursor2 = conn2.cursor()
            cursor2.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES (?, 'PASSWORD_RESET', ?, ?)
            ''', (u['id'], 'Password reset via reset code', datetime.now()))
            conn2.commit()
            conn2.close()
        except Exception:
            pass
        
        return jsonify(success=True, message='密码重置成功，请使用新密码登录')
    except Exception as e:
        return jsonify(success=False, message=f'重置失败: {e}')


@app.route('/api/permissions/my')
def get_my_permissions():
    """获取当前用户的权限列表（前端用于 UI 控制）"""
    role = session.get('role', 'guest')
    permissions = PermissionManager.get_user_permissions(role)
    
    menu_items = []
    if role == 'user':
        menu_items = [
            {'name': '行情数据', 'path': '/market', 'icon': 'TrendCharts'},
            {'name': '特征工程', 'path': '/features', 'icon': 'DataAnalysis'},
            {'name': '模型预测', 'path': '/predictions', 'icon': 'Cpu'},
            {'name': '模拟交易', 'path': '/trading', 'icon': 'Money'},
            {'name': '风险管理', 'path': '/risk', 'icon': 'Warning'},
            {'name': '个人中心', 'path': '/profile', 'icon': 'User'}
        ]
    elif role == 'admin':
        menu_items = [
            {'name': '数据管理', 'path': '/admin/market', 'icon': 'Database'},
            {'name': '风控配置', 'path': '/admin/risk', 'icon': 'Warning'},
            {'name': '用户管理', 'path': '/admin/users', 'icon': 'UserFilled'},
            {'name': '系统审计', 'path': '/admin/audit', 'icon': 'Document'},
            {'name': '资源配额', 'path': '/admin/quota', 'icon': 'Setting'}
        ]
    
    return jsonify(
        success=True,
        role=role,
        permissions=permissions,
        menu=menu_items
    )


@app.route('/api/admin/dashboard')
@require_admin
def admin_dashboard_api():
    """管理员仪表盘统计数据（支持近N天统计与图表）"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        try:
            days = int(request.args.get('days', 30))
        except Exception:
            days = 30
        days = max(7, min(days, 365))
        since_dt = datetime.now() - timedelta(days=days - 1)
        since = since_dt.strftime('%Y-%m-%d 00:00:00')

        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE date(created_at)=date('now', 'localtime')")
        today_analyses = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM analysis_history')
        total_analyses_all = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM analysis_history WHERE created_at >= ?', (since,))
        total_analyses_in_range = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM trading_records')
        total_trades = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM analysis_history WHERE created_at >= ?', (since,))
        active_users = cursor.fetchone()[0] or 0
        cursor.execute('SELECT ROUND(AVG(execution_time), 2) FROM analysis_history WHERE execution_time IS NOT NULL AND created_at >= ?', (since,))
        avg_execution_time = cursor.fetchone()[0]

        labels = [(since_dt + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        default_map = {d: 0 for d in labels}

        cursor.execute('''
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM users
            WHERE created_at >= ?
            GROUP BY date(created_at)
            ORDER BY d ASC
        ''', (since,))
        user_growth_map = dict(default_map)
        for row in cursor.fetchall():
            user_growth_map[str(row[0])] = int(row[1] or 0)

        cursor.execute('''
            SELECT date(created_at) AS d, COUNT(*) AS c
            FROM analysis_history
            WHERE created_at >= ?
            GROUP BY date(created_at)
            ORDER BY d ASC
        ''', (since,))
        analysis_trend_map = dict(default_map)
        for row in cursor.fetchall():
            analysis_trend_map[str(row[0])] = int(row[1] or 0)

        cursor.execute('''
            SELECT COALESCE(analysis_type, '未知') AS analysis_type, COUNT(*) AS c
            FROM analysis_history
            WHERE created_at >= ?
            GROUP BY analysis_type
            ORDER BY c DESC
            LIMIT 8
        ''', (since,))
        analysis_type_dist = [{'name': str(r[0]), 'value': int(r[1] or 0)} for r in cursor.fetchall()]

        cursor.execute('''
            SELECT COALESCE(u.username, '未知用户') AS username, COUNT(a.id) AS c
            FROM analysis_history a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE a.created_at >= ?
            GROUP BY a.user_id, username
            ORDER BY c DESC
            LIMIT 8
        ''', (since,))
        top_users_rows = cursor.fetchall()
        top_users = {
            'labels': [str(r[0]) for r in top_users_rows],
            'values': [int(r[1] or 0) for r in top_users_rows],
        }

        stats = {
            'totalUsers': total_users,
            'todayAnalyses': today_analyses,
            'totalAnalyses': total_analyses_all,
            'rangeAnalyses': total_analyses_in_range,
            'totalTrades': total_trades,
            'activeUsers': active_users,
            'avgExecutionTime': float(avg_execution_time) if avg_execution_time is not None else None,
            'systemStatus': '正常'
        }
        charts = {
            'range_days': days,
            'user_growth': {'labels': labels, 'values': [user_growth_map[d] for d in labels]},
            'analysis_trend': {'labels': labels, 'values': [analysis_trend_map[d] for d in labels]},
            'analysis_type_dist': analysis_type_dist,
            'top_users': top_users,
        }
        return jsonify(success=True, stats=stats, charts=charts)
    except Exception as e:
        return jsonify(success=False, message=f'获取仪表盘数据失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/users')
@require_admin
def admin_users():
    """管理员获取用户列表"""
    users = db['user_manager'].get_all_users()
    return jsonify(success=True, users=users)


@app.route('/api/admin/users/<int:user_id>/role', methods=['POST'])
@require_admin
def admin_update_user_role(user_id: int):
    """管理员修改用户角色"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        d = request.get_json() or {}
        role = (d.get('role') or '').strip()
        admin_id = int(session.get('user_id') or 0)
        valid_roles = set(db['role_permission_manager'].get_roles() or ['admin', 'user', 'guest'])
        if role not in valid_roles:
            return jsonify(success=False, message='无效角色')

        cursor.execute('SELECT id, role, username FROM users WHERE id = ?', (user_id,))
        target = cursor.fetchone()
        if not target:
            return jsonify(success=False, message='目标用户不存在')

        if admin_id == user_id and role != 'admin':
            return jsonify(success=False, message='不能将当前登录管理员降级')

        cursor.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (?, 'ADMIN_UPDATE_ROLE', ?, ?, ?)
        ''', (admin_id, f'user={target["username"]}, new_role={role}', request.remote_addr, datetime.now()))
        conn.commit()
        return jsonify(success=True, message='角色更新成功')
    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=f'角色更新失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/users/<int:user_id>/reset_password', methods=['POST'])
@require_admin
def admin_reset_user_password(user_id: int):
    """管理员重置用户密码"""
    try:
        d = request.get_json() or {}
        new_password = (d.get('new_password') or '').strip()
        if len(new_password) < 6:
            return jsonify(success=False, message='密码至少6位')
        admin_id = int(session.get('user_id') or 0)
        ok = db['user_manager'].admin_reset_user_password(admin_id, user_id, new_password)
        return jsonify(success=ok, message='重置成功' if ok else '重置失败')
    except Exception as e:
        return jsonify(success=False, message=f'重置失败: {e}')


@app.route('/api/admin/users/<int:user_id>/toggle', methods=['POST'])
@require_admin
def admin_toggle_user(user_id: int):
    """管理员冻结/启用用户"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        admin_id = int(session.get('user_id') or 0)
        cursor.execute('SELECT id, username, is_active FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify(success=False, message='目标用户不存在')
        if admin_id == user_id and int(row['is_active'] or 0) == 1:
            return jsonify(success=False, message='不能禁用当前管理员')

        new_state = 0 if int(row['is_active'] or 0) == 1 else 1
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_state, user_id))
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (?, 'ADMIN_TOGGLE_USER', ?, ?, ?)
        ''', (admin_id, f'user={row["username"]}, is_active={new_state}', request.remote_addr, datetime.now()))
        conn.commit()
        return jsonify(success=True, message='已更新用户状态', is_active=bool(new_state))
    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=f'操作失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@require_admin
def admin_delete_user(user_id: int):
    """管理员删除用户及关联数据"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        admin_id = int(session.get('user_id') or 0)
        if admin_id == user_id:
            return jsonify(success=False, message='不能删除当前登录管理员')

        cursor.execute('SELECT id, username, role FROM users WHERE id = ?', (user_id,))
        target = cursor.fetchone()
        if not target:
            return jsonify(success=False, message='目标用户不存在')

        if str(target['role']) == 'admin':
            cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1")
            if int(cursor.fetchone()[0] or 0) <= 1:
                return jsonify(success=False, message='至少保留一个活跃管理员')

        for sql in [
            'DELETE FROM trading_records WHERE user_id = ?',
            'DELETE FROM analysis_history WHERE user_id = ?',
            'DELETE FROM user_sessions WHERE user_id = ?',
            'DELETE FROM user_favorites WHERE user_id = ?',
            'DELETE FROM user_feedback WHERE user_id = ?',
            'DELETE FROM portfolios WHERE user_id = ?',
            'DELETE FROM user_transactions WHERE user_id = ?',
            'DELETE FROM user_settings WHERE user_id = ?',
            'DELETE FROM password_resets WHERE user_id = ?',
            'DELETE FROM admin_task_runs WHERE user_id = ?',
            'DELETE FROM audit_logs WHERE user_id = ?',
        ]:
            cursor.execute(sql, (user_id,))
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
            VALUES (?, 'ADMIN_DELETE_USER', ?, ?, ?)
        ''', (admin_id, f'delete_user={target["username"]}', request.remote_addr, datetime.now()))
        conn.commit()
        return jsonify(success=True, message='用户已删除')
    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=f'删除失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/password_resets')
@require_admin
def admin_password_resets():
    """管理员查看最近的密码重置记录"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT pr.id, u.username, pr.reset_code, pr.created_at, pr.expires_at, pr.is_used
            FROM password_resets pr
            LEFT JOIN users u ON pr.user_id = u.id
            ORDER BY pr.created_at DESC
            LIMIT 100
        ''')
        rows = cursor.fetchall()
        resets = []
        for r in rows:
            resets.append({
                'id': r[0],
                'username': r[1],
                'reset_code': r[2],
                'created_at': r[3],
                'expires_at': r[4],
                'is_used': bool(r[5]),
            })
        return jsonify(success=True, resets=resets)
    except Exception as e:
        return jsonify(success=False, message=f'获取重置码记录失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/analyses')
@require_admin
def admin_analyses():
    """管理员查看分析记录（简化版）"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT a.id, u.username, a.ticker, a.analysis_type, a.created_at, a.execution_time
            FROM analysis_history a
            LEFT JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC
            LIMIT 200
        ''')
        rows = cursor.fetchall()
        analyses = []
        for r in rows:
            analyses.append({
                'id': r[0],
                'username': r[1],
                'ticker': r[2],
                'analysis_type': r[3],
                'created_at': r[4],
                'execution_time': r[5],
            })
        return jsonify(success=True, analyses=analyses)
    except Exception as e:
        return jsonify(success=False, message=f'获取分析记录失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/analyses/<int:analysis_id>', methods=['DELETE'])
@require_admin
def admin_delete_analysis(analysis_id: int):
    """管理员删除分析记录"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM analysis_history WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify(success=False, message='分析记录不存在')
        cursor.execute('DELETE FROM trading_records WHERE analysis_id = ?', (analysis_id,))
        cursor.execute('DELETE FROM analysis_history WHERE id = ?', (analysis_id,))
        conn.commit()
        return jsonify(success=True, message='分析记录已删除')
    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=f'删除失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/trades')
@require_admin
def admin_trades():
    """管理员查看交易记录（不含敏感盈亏信息）"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT t.id, u.username, t.ticker, t.action, t.quantity, t.price, t.timestamp, t.strategy_used
            FROM trading_records t
            LEFT JOIN users u ON t.user_id = u.id
            ORDER BY t.timestamp DESC
            LIMIT 200
        ''')
        rows = cursor.fetchall()
        trades = []
        for r in rows:
            trades.append({
                'id': r[0],
                'username': r[1],
                'ticker': r[2],
                'action': r[3],
                'quantity': r[4],
                'price': r[5],
                'created_at': r[6],
                'strategy': r[7],
            })
        return jsonify(success=True, trades=trades)
    except Exception as e:
        return jsonify(success=False, message=f'获取交易记录失败: {e}')
    finally:
        conn.close()


@app.route('/api/admin/system')
@require_admin
def admin_system_info():
    """管理员查看系统资源信息（演示用静态数据）"""
    # 为展示方便，这里返回静态模拟数据，实际可接入 psutil 等库获取实时信息
    system_info = {
        'cpuUsage': 35,
        'memoryUsage': 62,
        'diskUsage': 48,
        'uptime': '1天5小时'
    }
    return jsonify(success=True, systemInfo=system_info)


# ---------- 角色管理 ----------
@app.route('/api/admin/roles')
@require_admin
def admin_roles():
    """获取角色及权限配置"""
    try:
        rpm = db['role_permission_manager']
        roles_perms = rpm.get_all_roles_permissions()
        labels = rpm.get_permission_labels()
        roles = rpm.get_roles()
        return jsonify(success=True, roles=roles, rolePermissions=roles_perms, permissionLabels=labels)
    except Exception as e:
        return jsonify(success=False, message=str(e))


@app.route('/api/admin/roles/permissions', methods=['POST'])
@require_admin
def admin_update_role_permission():
    """更新角色权限"""
    try:
        d = request.get_json() or {}
        role = d.get('role')
        permission = d.get('permission')
        enabled = d.get('enabled', True)
        if not role or not permission:
            return jsonify(success=False, message='缺少 role 或 permission')
        rpm = db['role_permission_manager']
        ok = rpm.set_permission(role, permission, enabled)
        return jsonify(success=ok, message='更新成功' if ok else '更新失败')
    except Exception as e:
        return jsonify(success=False, message=str(e))


# ---------- 高级设置（模型/策略/参数限制） ----------
DEFAULT_ADVANCED = {
    'multi_model_enabled': True,
    'available_strategies': ['momentum', 'mean_reversion', 'rsi', 'macd', 'ma_crossover', 'breakout'],
    'epochs_max': 2000,
    'epochs_min': 50,
    'learning_rate_min': 0.0001,
    'learning_rate_max': 0.01,
    'initial_money_max': 1000000,
    'initial_money_min': 1000,
}


@app.route('/api/admin/advanced_settings')
@require_admin
def admin_advanced_settings_get():
    """获取高级设置"""
    try:
        conn = db['db']
        raw = conn.get_config('advanced_settings', '{}')
        if not raw:
            return jsonify(success=True, settings=dict(DEFAULT_ADVANCED))
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            data = dict(DEFAULT_ADVANCED)
        for k, v in DEFAULT_ADVANCED.items():
            if k not in data:
                data[k] = v
        return jsonify(success=True, settings=data)
    except Exception as e:
        return jsonify(success=False, message=str(e))


@app.route('/api/admin/advanced_settings', methods=['POST'])
@require_admin
def admin_advanced_settings_save():
    """保存高级设置"""
    try:
        d = request.get_json() or {}
        conn = db['db']
        conn.set_config('advanced_settings', json.dumps(d))
        return jsonify(success=True, message='高级设置已保存')
    except Exception as e:
        return jsonify(success=False, message=str(e))


# ---------- 系统设置（兼容现有前端） ----------
@app.route('/api/admin/settings', methods=['GET', 'POST'])
@require_admin
def admin_settings():
    """系统基础设置"""
    try:
        conn = db['db']
        if request.method == 'GET':
            raw = conn.get_config('admin_settings', '{}')
            try:
                data = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                data = {}
            defaults = {
                'systemName': '智能股票交易系统',
                'maxUsers': 1000,
                'updateInterval': 300,
                'emailNotification': True,
                'testEmail': ''
            }
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            return jsonify(success=True, settings=data)
        else:
            d = request.get_json() or {}
            conn.set_config('admin_settings', json.dumps(d))
            return jsonify(success=True, message='设置已保存')
    except Exception as e:
        return jsonify(success=False, message=str(e))


@app.route('/api/admin/send_test_email', methods=['POST'])
@require_admin
def admin_send_test_email():
    """测试邮件发送（未配置SMTP则模拟成功，避免前端报错）"""
    try:
        d = request.get_json() or {}
        to_email = (d.get('to_email') or '').strip()
        if not to_email:
            return jsonify(success=False, message='请填写测试邮箱')
        if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', to_email):
            return jsonify(success=False, message='邮箱格式不正确')
        return jsonify(success=True, message='测试邮件已发送（演示模式）')
    except Exception as e:
        return jsonify(success=False, message=f'发送失败: {e}')


@app.route('/api/admin/feedbacks')
@require_admin
def admin_feedbacks():
    """管理员查看用户反馈（联表返回用户名）"""
    try:
        status = (request.args.get('status') or 'all').strip()
        limit = int(request.args.get('limit') or 200)
    except Exception:
        status, limit = 'all', 200
    rows = db['feedback_manager'].get_all_feedback(status=status, limit=limit)
    return jsonify(success=True, feedbacks=rows)


@app.route('/api/admin/feedbacks/<int:feedback_id>/reply', methods=['POST'])
@require_admin
def admin_feedback_reply(feedback_id: int):
    """管理员回复用户反馈"""
    try:
        d = request.get_json() or {}
        reply_text = (d.get('admin_reply') or '').strip()
        status = (d.get('status') or 'resolved').strip()
        if not reply_text:
            return jsonify(success=False, message='回复内容不能为空')
        if status not in {'pending', 'processing', 'resolved', 'rejected'}:
            status = 'resolved'
        ok = db['feedback_manager'].reply_feedback(feedback_id, reply_text, status)
        return jsonify(success=ok, message='回复成功' if ok else '回复失败')
    except Exception as e:
        return jsonify(success=False, message=f'回复失败: {e}')


# ---------- 任务监控 ----------
@app.route('/api/admin/tasks')
@require_admin
def admin_tasks():
    """任务监控：支持按用户筛选任务与汇总统计"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        username = (request.args.get('username') or '').strip()
        try:
            days = int(request.args.get('days', 30))
        except Exception:
            days = 30
        try:
            limit = int(request.args.get('limit', 200))
        except Exception:
            limit = 200

        days = max(1, min(days, 3650))
        limit = max(10, min(limit, 500))
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

        where_sql = "a.created_at >= ?"
        params = [since]
        if username:
            where_sql += " AND u.username = ?"
            params.append(username)

        cursor.execute(f'''
            SELECT a.id, a.user_id, COALESCE(u.username, '未知用户') AS username,
                   a.ticker, a.analysis_type, a.created_at, a.execution_time
            FROM analysis_history a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE {where_sql}
            ORDER BY a.created_at DESC
            LIMIT ?
        ''', tuple(params + [limit]))

        rows = cursor.fetchall()
        tasks = []
        for r in rows:
            exec_time = r[6]
            exec_time = round(float(exec_time), 2) if exec_time is not None else None
            tasks.append({
                'id': r[0],
                'user_id': r[1],
                'username': r[2],
                'ticker': r[3],
                'type': r[4],
                'created_at': r[5],
                'execution_time': exec_time,
                'status': '已完成'
            })

        # 用户下拉选项
        cursor.execute("SELECT username FROM users ORDER BY username ASC")
        available_users = [row[0] for row in cursor.fetchall() if row[0]]

        summary = {
            'total': len(tasks),
            'completed': len(tasks),
            'failed': 0,
            'running': 0,
            'user_count': len({t['username'] for t in tasks if t.get('username')}),
            'range_days': days
        }
        return jsonify(
            success=True,
            tasks=tasks,
            summary=summary,
            filters={'username': username, 'days': days, 'limit': limit},
            available_users=available_users
        )
    except Exception as e:
        return jsonify(success=False, message=str(e))
    finally:
        conn.close()


@app.route('/api/admin/tasks/user_overview')
@require_admin
def admin_user_task_overview():
    """用户任务概览：按用户统计任务数量、最近任务时间、平均耗时"""
    conn = db['db'].get_connection()
    cursor = conn.cursor()
    try:
        try:
            days = int(request.args.get('days', 30))
        except Exception:
            days = 30
        days = max(1, min(days, 3650))
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            SELECT
                u.id,
                u.username,
                u.is_active,
                COUNT(a.id) AS total_tasks,
                SUM(CASE WHEN date(a.created_at) = date('now', 'localtime') THEN 1 ELSE 0 END) AS today_tasks,
                MAX(a.created_at) AS last_task_at,
                ROUND(AVG(CASE WHEN a.execution_time IS NOT NULL THEN a.execution_time END), 2) AS avg_execution_time
            FROM users u
            LEFT JOIN analysis_history a
                ON a.user_id = u.id
               AND a.created_at >= ?
            GROUP BY u.id, u.username, u.is_active
            ORDER BY total_tasks DESC, today_tasks DESC, u.id ASC
            LIMIT 300
        ''', (since,))

        overview = []
        total_tasks = 0
        active_task_users = 0
        for r in cursor.fetchall():
            total_count = int(r[3] or 0)
            total_tasks += total_count
            if total_count > 0:
                active_task_users += 1
            overview.append({
                'user_id': r[0],
                'username': r[1],
                'status': 'active' if int(r[2] or 0) == 1 else 'inactive',
                'total_tasks': total_count,
                'today_tasks': int(r[4] or 0),
                'last_task_at': r[5],
                'avg_execution_time': r[6]
            })

        summary = {
            'days': days,
            'user_count': len(overview),
            'active_task_users': active_task_users,
            'total_tasks': total_tasks
        }
        return jsonify(success=True, overview=overview, summary=summary)
    except Exception as e:
        return jsonify(success=False, message=str(e))
    finally:
        conn.close()


# ---------- 市场数据 ----------
@app.route('/api/market/available')
def market_available():
    avail = set()
    for root in ['data', 'stock_trading/data']:
        if os.path.isdir(root):
            avail.update(f.replace('.csv', '') for f in os.listdir(root) if f.endswith('.csv'))
    
    # 添加A股热门股票
    a_stocks = ['600519', '000001', '300750', '600036', '000858', '002415', '600276', '000002', '600031', '002594']
    avail.update(a_stocks)
    
    return jsonify(success=True, stocks=sorted(avail))

@app.route('/api/market/a_stocks')
def a_stocks_list():
    """获取A股股票列表"""
    try:
        stock_list = get_stock_list()
        if not stock_list.empty:
            return jsonify(success=True, stocks=stock_list.to_dict('records'))
        else:
            # 返回热门A股作为备选
            hot_stocks = [
                {'代码': '600519', '名称': '贵州茅台', '最新价': 0, '涨跌幅': 0, '成交量': 0, '成交额': 0},
                {'代码': '000001', '名称': '平安银行', '最新价': 0, '涨跌幅': 0, '成交量': 0, '成交额': 0},
                {'代码': '300750', '名称': '宁德时代', '最新价': 0, '涨跌幅': 0, '成交量': 0, '成交额': 0},
                {'代码': '600036', '名称': '招商银行', '最新价': 0, '涨跌幅': 0, '成交量': 0, '成交额': 0},
                {'代码': '000858', '名称': '五粮液', '最新价': 0, '涨跌幅': 0, '成交量': 0, '成交额': 0}
            ]
            return jsonify(success=True, stocks=hot_stocks)
    except Exception as e:
        return jsonify(success=False, message=f'获取A股列表失败: {e}')

# 实时搜索（A股 + 美股）
@app.route('/api/market/search')
def market_search():
    try:
        q = (request.args.get('q') or '').strip()
        limit = int(request.args.get('limit') or 20)
        results = { 'a_stocks': [], 'us_stocks': [] }

        # A股：调用 get_stock_list 并进行关键词过滤（代码/名称）
        try:
            a_df = get_stock_list()
            if not a_df.empty:
                if q:
                    mask = a_df['代码'].astype(str).str.contains(q, case=False) | a_df['名称'].astype(str).str.contains(q, case=False)
                    a_df = a_df[mask]
                a_df = a_df.head(limit)
                results['a_stocks'] = [
                    {
                        'code': str(row['代码']),
                        'name': str(row['名称']),
                        'price': float(row.get('最新价') or 0) if pd.notnull(row.get('最新价')) else 0.0,
                        'change': float(row.get('涨跌幅') or 0) if pd.notnull(row.get('涨跌幅')) else 0.0,
                        'market': 'CN'
                    } for _, row in a_df.iterrows()
                ]
        except Exception as e:
            print(f"A股搜索失败，使用备选方案: {e}")
            pass
        
        # 备选方案：基于本地数据文件（总是执行）
        try:
            a_stock_files = []
            for root in ['data', 'stock_trading/data']:
                if os.path.isdir(root):
                    for f in os.listdir(root):
                        if f.endswith('.csv') and len(f.replace('.csv', '')) == 6 and f.replace('.csv', '').isdigit():
                            a_stock_files.append(f.replace('.csv', ''))
            
            # 添加热门A股作为备选
            hot_a_stocks = ['600519', '000001', '300750', '600036', '000858', '002415', '600276', '000002', '600031', '002594']
            all_a_stocks = list(set(a_stock_files + hot_a_stocks))
            
            # 过滤匹配的A股代码
            if q:
                all_a_stocks = [code for code in all_a_stocks if q in code]
            
            all_a_stocks = all_a_stocks[:limit]
            
            # 如果A股结果为空，使用备选结果
            if not results['a_stocks'] and all_a_stocks:
                results['a_stocks'] = [
                    {
                        'code': code,
                        'name': f'A股{code}',
                        'price': 0.0,
                        'change': 0.0,
                        'market': 'CN'
                    } for code in all_a_stocks
                ]
        except Exception as e2:
            print(f"A股备选方案也失败: {e2}")
            pass

        # 美股：基于本地可用清单 + 关键字过滤（代码匹配）
        try:
            avail = set()
            for root in ['stock_trading/data', 'data']:
                if os.path.isdir(root):
                    avail.update(f.replace('.csv', '') for f in os.listdir(root) if f.endswith('.csv'))
            us = sorted([s for s in avail if s.isalpha() and (q.upper() in s if q else True)])[:limit]
            results['us_stocks'] = [
                { 'code': s, 'name': s, 'price': 0, 'change': 0, 'market': 'US' } for s in us
            ]
        except Exception:
            pass

        return jsonify(success=True, results=results)
    except Exception as e:
        return jsonify(success=False, message=f'搜索失败: {e}')

@app.route('/api/market/overview')
def market_overview():
    return jsonify(success=True, overview={'total_stocks': 30, 'market_status': 'open'})

@app.route('/api/market/sectors')
def market_sectors():
    return jsonify(success=True, sectors=['科技', '金融', '医疗', '能源', '消费'])

@app.route('/api/market/realtime/<ticker>')
def realtime_data(ticker):
    try:
        data = market_api.get_realtime_data(ticker.upper())
        return jsonify(success=True, data=data)
    except Exception as e:
        return jsonify(success=False, message=str(e))

# ---------- 分析功能 ----------
@app.route('/api/analysis/run', methods=['POST'])
@login_required
def run_analysis():
    """
    完整的股票分析功能
    合并数据获取 + 模型训练 + 交易分析
    """
    try:
        d = request.json or {}
        ticker = d['ticker'].upper()
        
        # 1. 获取数据
        csv_path = _ensure_ticker_data(
            ticker,
            d.get('start_date', '2020-01-01'),
            d.get('end_date', str(datetime.today().date()))
        )
        if not csv_path: 
            return jsonify(success=False, message='本地无数据')
        
        # 2. 特征工程
        df = pd.read_csv(csv_path, parse_dates=True, index_col='Date')
        X, y = format_feature(df)
        
        # 3. 训练模型 & 交易分析
        metrics = predict(ticker_name=ticker, stock_data=df, stock_features=(X, y), save_dir=SAVE_DIR,
                          epochs=d.get('epochs', 200), batch_size=d.get('batch_size', 32),
                          learning_rate=d.get('learning_rate', 0.001))
        trading = process_stock(ticker, SAVE_DIR, window_size=d.get('window_size', 30),
                                initial_money=d.get('initial_money', 10000), iterations=d.get('agent_iterations', 300))
        
        # 4. 读取交易记录
        trans = pd.read_csv(f'{SAVE_DIR}/transactions/{ticker}_transactions.csv')
        
        # 5. 保存分析结果（游客不入库）
        analysis_id = None
        if session.get('guest') is not True and 'user_id' in session:
            analysis_id = db['analysis_manager'].save_analysis(
            session['user_id'], ticker, 'LSTM_RL', d, metrics | trading, 0)
        
        return jsonify(success=True, analysis_id=analysis_id,
                       prediction_metrics={'accuracy': metrics['accuracy']*100, 'rmse': metrics['rmse'], 'mae': metrics['mae']},
                       trading_results=trading,
                       transactions=trans.to_dict('records'),
                       images={'prediction': f'/images/predictions/{ticker}_prediction.png',
                               'loss': f'/images/loss/{ticker}_loss.png',
                               'earnings': f'/images/earnings/{ticker}_cumulative.png',
                               'trades': f'/images/trades/{ticker}_trades.png'})
    except Exception as e:
        return jsonify(success=False, message=f'分析失败：{e}')

# ---------- 兼容5目录成熟API：数据裁剪与训练 ----------
@app.route('/get_data', methods=['POST', 'GET', 'OPTIONS'])
@app.route('/api/get_data', methods=['POST', 'GET', 'OPTIONS'])
def get_data():
    try:
        # 允许GET快速探活：/get_data?ticker=600519&start_date=2020-01-01&end_date=2020-12-31
        if request.method == 'GET':
            d = request.args or {}
        else:
            d = request.get_json() or {}
        ticker = d['ticker']
        start_date = d['start_date']
        end_date = d['end_date']
        model_type = d.get('model_type', 'lstm')

        # 智能数据源选择：优先本地缓存，其次实时获取
        csv_path = None
        
        # 1. 优先查找本地文件（包含完整特征的数据）
        for base in ['stock_trading/data', 'data']:
            p = os.path.join(base, f'{ticker}.csv')
            if os.path.isfile(p):
                csv_path = p
                print(f"使用本地缓存: {csv_path}")
                break
        
        # 2. 如果本地没有数据，才进行实时获取
        if not csv_path:
            if validate_stock_code(ticker):
                try:
                    print(f"检测到A股代码 {ticker}，使用AKShare实时获取...")
                    csv_path = download_a_share(ticker, start_date, end_date, out_dir='data')
                    
                    # A股数据需要特征工程处理
                    if csv_path and os.path.exists(csv_path):
                        print(f"A股数据获取成功，开始特征工程...")
                        try:
                            # 对A股数据进行特征工程处理
                            enhanced_path = process_a_share_data(csv_path)
                            csv_path = enhanced_path
                            print(f"A股特征工程完成: {enhanced_path}")
                        except Exception as e:
                            print(f"A股特征工程失败: {e}，使用原始数据")
                        
                except Exception as e:
                    print(f"AKShare获取失败: {e}")
            else:
                print(f"美股代码 {ticker}，使用 yfinance 实时获取...")
                try:
                    csv_path = download_us_stock(ticker, start_date, end_date, out_dir='data')
                    if csv_path and os.path.exists(csv_path):
                        print(f"美股数据获取成功，开始特征工程...")
                        try:
                            enhanced_path = process_a_share_data(csv_path)
                            csv_path = enhanced_path
                            print(f"美股特征工程完成: {enhanced_path}")
                        except Exception as e:
                            print(f"美股特征工程失败: {e}，使用原始数据")
                except Exception as e:
                    print(f"yfinance 获取失败: {e}")
        
        if not csv_path:
            return jsonify(success=False, message=f'未找到 {ticker} 的数据文件，请检查股票代码或先下载数据')

        # 3. 读取并处理数据
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 日期范围过滤 - 处理时区问题
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        
        # 统一处理时区问题
        if df['Date'].dt.tz is not None:
            # 数据有时区信息，将查询日期转换为相同时区
            if s.tz is None: 
                s = s.tz_localize('UTC')
            if e.tz is None: 
                e = e.tz_localize('UTC')
        else:
            # 数据没有时区信息，将查询日期也去掉时区
            if s.tz is not None: 
                s = s.tz_localize(None)
            if e.tz is not None: 
                e = e.tz_localize(None)

        fdf = df[(df['Date'] >= s) & (df['Date'] <= e)]
        
        # 调试信息
        print(f"原始数据: {len(df)} 行")
        print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")
        print(f"查询范围: {s} 到 {e}")
        print(f"过滤后数据: {len(fdf)} 行")
        
        min_rows = 37 if model_type == 'lightweight' else 100
        if len(fdf) < min_rows:
            return jsonify(success=False, message=f'数据量不足，仅 {len(fdf)} 行，{"轻量模型" if model_type == "lightweight" else "LSTM"}至少需要{min_rows}行')

        # 4. 保存到临时目录供后续分析使用
        tmp = os.path.join(TEMP_DIR, 'ticker')
        os.makedirs(tmp, exist_ok=True)
        out = os.path.join(tmp, f'{ticker}.csv')
        
        # 检查数据是否包含必要的技术指标
        required_features = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR', 'Close_yes', 'Open_yes', 'High_yes', 'Low_yes']
        missing_features = [col for col in required_features if col not in fdf.columns]
        
        if missing_features:
            print(f"数据缺少技术指标: {missing_features}")
            
            # 如果是A股数据，进行特征工程
            if validate_stock_code(ticker):
                print(f"A股数据缺少技术指标，进行特征工程...")
                try:
                    # 临时保存原始数据
                    temp_file = os.path.join(tmp, f'{ticker}_temp.csv')
                    fdf.to_csv(temp_file, index=False)
                    
                    # 进行特征工程
                    enhanced_file = process_a_share_data(temp_file)
                    
                    # 读取增强后的数据
                    fdf_enhanced = pd.read_csv(enhanced_file)
                    fdf_enhanced.to_csv(out, index=False)
                    
                    # 清理临时文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    if os.path.exists(enhanced_file):
                        os.remove(enhanced_file)
                        
                    print(f"A股特征工程完成，保存到: {out}")
                    # 更新fdf为增强后的数据
                    fdf = fdf_enhanced
                except Exception as e:
                    print(f"A股特征工程失败: {e}，使用原始数据")
                    fdf.to_csv(out, index=False)
            else:
                # 美股数据缺少特征，返回错误
                return jsonify(success=False, message=f'美股数据缺少必要的技术指标: {missing_features}，请检查数据文件')
        else:
            fdf.to_csv(out, index=False)
        
        # 5. 返回成功信息
        market_type = "A股" if validate_stock_code(ticker) else "美股"
        return jsonify(success=True, message=f'{market_type}数据加载成功: {ticker} ({len(fdf)} 行)', file_path=out)
    except Exception as e:
        return jsonify(success=False, message=f'加载数据出错: {e}')

@app.route('/train_model', methods=['POST'])
@app.route('/api/train_model', methods=['POST'])
def train_model():
    try:
        # 性能分析开始
        total_start = ts()
        print(f"\n🚀 开始分析 {request.get_json().get('ticker', 'UNKNOWN')}")
        
        d = request.get_json() or {}
        ticker = d['ticker'].upper()
        model_type = d.get('model_type', 'lstm')  # lstm | lightweight
        epochs = d.get('epochs', 500)
        batch_size = d.get('batch_size', 32)
        learning_rate = d.get('learning_rate', 0.001)
        window_size = d.get('window_size', 30)
        initial_money = d.get('initial_money', 10000)
        agent_iterations = d.get('agent_iterations', 500)
        demo_mode = d.get('demo', False)  # 演示模式

        # 演示模式：直接返回预生成结果
        if demo_mode:
            print("🎭 演示模式：返回预生成结果")
            demo_result = {
                'success': True,
                'prediction_metrics': {'accuracy': 68.2, 'rmse': 0.021, 'mae': 0.017},
                'trading_results': {'total_return': 0.214, 'max_drawdown': -0.078, 'sharpe_ratio': 1.34, 'trades_buy': 45, 'trades_sell': 45},
                'transactions': [{'day': i, 'operate': 'buy' if i%2==0 else 'sell', 'price': 150+i, 'total_balance': 10000+i*10} for i in range(20)],
                'images': {
                    'prediction': f"/images/predictions/{ticker}_prediction.png",
                    'loss': f"/images/loss/{ticker}_loss.png", 
                    'earnings': f"/images/earnings/{ticker}_cumulative.png",
                    'trades': f"/images/trades/{ticker}_trades.png"
                }
            }
            print(f"🎯 演示模式总耗时：{ts()-total_start:.2f}s")
            return jsonify(demo_result)

        # ① 数据 IO 阶段
        t0 = ts()
        temp_csv = os.path.join(TEMP_DIR, 'ticker', f'{ticker}.csv')
        if not os.path.isfile(temp_csv):
            return jsonify(success=False, message='请先获取股票数据')

        stock_data = pd.read_csv(temp_csv)
        n_rows = len(stock_data)
        # 数据 37-99 行时强制使用轻量模型（防止前端未正确传 model_type）
        if 37 <= n_rows < 100 and model_type != 'lightweight':
            print(f"⚠ 数据 {n_rows} 行，自动切换为轻量模型")
            model_type = 'lightweight'
        if model_type == 'lstm':
            if n_rows < 100:
                return jsonify(success=False, message=f'LSTM 模型需要至少 100 行数据，当前仅 {n_rows} 行，请选择「轻量模型」或增加数据')
        else:
            if n_rows < 37:
                return jsonify(success=False, message=f'数据量不足，仅 {n_rows} 行，轻量模型至少需要 37 行')
        if 'Date' in stock_data.columns:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)

        # 检查数据是否包含必要的特征列
        required_features = [
            'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
            'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
            'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
        ]
        
        missing_features = [f for f in required_features if f not in stock_data.columns]
        if missing_features:
            print(f"缺少特征列: {missing_features}")
            # 尝试从原始数据重新生成特征
            try:
                from a_share_feature_engineering import process_a_share_data
                temp_path = os.path.join(TEMP_DIR, 'ticker', f'{ticker}_temp.csv')
                stock_data.reset_index().to_csv(temp_path, index=False)
                enhanced_path = process_a_share_data(temp_path)
                stock_data = pd.read_csv(enhanced_path)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                print(f"重新生成特征完成: {enhanced_path}")
            except Exception as e:
                print(f"重新生成特征失败: {e}")
                return jsonify(success=False, message=f'数据缺少必要特征: {missing_features[:5]}...')
        
        print(f"① 数据 IO：{ts()-t0:.2f}s")
        
        # ② 特征工程阶段
        t0 = ts()
        X, y = format_feature(stock_data)
        print(f"② 特征工程：{ts()-t0:.2f}s")
        # ③ LSTM/轻量模型 训练阶段
        t0 = ts()
        if model_type == 'lightweight':
            n_steps, train_epochs, train_batch = 15, min(epochs, 150), min(batch_size, 16)
            rl_window = 5
        else:
            n_steps, train_epochs, train_batch = 60, epochs, batch_size
            rl_window = window_size
        metrics = predict(
            ticker_name=ticker,
            stock_data=stock_data,
            stock_features=(X, y),
            save_dir=SAVE_DIR,
            epochs=train_epochs,
            batch_size=train_batch,
            learning_rate=learning_rate,
            n_steps=n_steps,
        )
        print(f"③ {'轻量' if model_type == 'lightweight' else 'LSTM'} 训练：{ts()-t0:.2f}s")
        
        # ④ RL 交易阶段
        t0 = ts()
        trading = process_stock(
            ticker,
            SAVE_DIR,
            window_size=rl_window,
            initial_money=initial_money,
            iterations=agent_iterations,
        )
        print(f"④ RL 交易：{ts()-t0:.2f}s")
        
        trans = pd.read_csv(f"{SAVE_DIR}/transactions/{ticker}_transactions.csv")

        result = dict(
            success=True,
            prediction_metrics={'accuracy': metrics['accuracy'] * 100, 'rmse': metrics['rmse'], 'mae': metrics['mae']},
            trading_results=trading,
            transactions=trans.to_dict('records'),
            images={'prediction': f"/images/predictions/{ticker}_prediction.png", 'loss': f"/images/loss/{ticker}_loss.png", 'earnings': f"/images/earnings/{ticker}_cumulative.png", 'trades': f"/images/trades/{ticker}_trades.png"},
        )
        
        # 性能分析结束
        total_time = ts() - total_start
        print(f"🎯 总耗时：{total_time:.2f}s")
        print(f"📊 性能分析：数据IO({((ts()-total_start-total_time)/total_time*100):.1f}%) | 特征工程({((ts()-total_start-total_time)/total_time*100):.1f}%) | LSTM训练({((ts()-total_start-total_time)/total_time*100):.1f}%) | RL交易({((ts()-total_start-total_time)/total_time*100):.1f}%)")

        if 'user_id' in session and session.get('guest') is not True:
            try:
                atype = 'Lightweight_RL' if model_type == 'lightweight' else 'LSTM_RL'
                analysis_id = db['analysis_manager'].save_analysis(
                    session['user_id'], ticker, atype, d, result, 0)
                result['analysis_id'] = analysis_id
            except Exception:
                pass

        return jsonify(result)
    except Exception as e:
        return jsonify(success=False, message=f'训练出错: {e}')

# ---------- 兼容5目录成熟API：策略回测/风险/仓位/止损 ----------
@app.route('/api/strategies/backtest', methods=['POST'])
@login_required
def api_backtest():
    try:
        d = request.get_json() or {}
        ticker = (d.get('ticker') or '').strip().upper()
        if not ticker:
            return jsonify(success=False, message='请提供合法的股票代码')

        requested = d.get('strategies', ['momentum', 'mean_reversion', 'rsi', 'macd']) or []
        requested = [str(s).strip() for s in requested if str(s).strip()]

        src = None
        for base in ['data', 'stock_trading/data']:
            p = os.path.join(base, f'{ticker}.csv')
            if os.path.isfile(p):
                src = p
                break
        if not src:
            return jsonify(success=False, message=f'找不到 {ticker} 数据，请先在数据目录准备对应CSV')

        df = pd.read_csv(src)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)

        strategy_catalog = {
            'momentum': MomentumStrategy(),
            'momentum strategy': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'mean reversion': MeanReversionStrategy(),
            'mean reversion strategy': MeanReversionStrategy(),
            'rsi': RSIStrategy(),
            'rsi strategy': RSIStrategy(),
            'macd': MACDStrategy(),
            'macd strategy': MACDStrategy(),
            'ma_crossover': MovingAverageCrossoverStrategy(),
            'ma crossover': MovingAverageCrossoverStrategy(),
            'moving average': MovingAverageCrossoverStrategy(),
            'moving average crossover': MovingAverageCrossoverStrategy(),
            'breakout': BreakoutStrategy(),
            'breakout strategy': BreakoutStrategy(),
        }

        ts = TradingSystem()
        added = 0
        for name in requested:
            key = name.lower()
            matched = strategy_catalog.get(key)
            if matched:
                ts.add_strategy(matched)
                added += 1

        if added == 0:
            return jsonify(success=False, message='未匹配到任何可用策略，请检查传入的策略名称')

        results = ts.run_backtest(df)
        if not results:
            return jsonify(success=False, message='策略回测无结果，请检查数据是否满足策略要求')

        formatted = {
            k: {
            'total_return': f"{v['total_return']:.2%}",
            'annual_return': f"{v['annual_return']:.2%}",
            'volatility': f"{v['volatility']:.2%}",
            'sharpe_ratio': f"{v['sharpe_ratio']:.3f}",
            'max_drawdown': f"{v['max_drawdown']:.2%}",
            }
            for k, v in results.items()
        }

        best_name, best_res = ts.get_best_strategy()
        best_payload = None
        if best_name and best_res:
            best_payload = {
                'name': best_name,
                'sharpe_ratio': best_res['sharpe_ratio']
            }

        return jsonify(success=True, results=formatted, best_strategy=best_payload)
    except Exception as e:
        return jsonify(success=False, message=f'策略回测失败: {e}')

@app.route('/api/risk_metrics', methods=['POST'])
@login_required
def api_risk_metrics():
    try:
        d = request.get_json() or {}
        transactions = d.get('transactions', [])
        if not transactions:
            return jsonify(success=False, message='没有交易记录')
        
        df = pd.DataFrame(transactions)
        print(f"交易记录列名: {list(df.columns)}")
        print(f"交易记录前3行:\n{df.head(3)}")
        
        # 确保数值列正确转换
        if 'total_balance' in df.columns:
            bal = pd.to_numeric(df['total_balance'], errors='coerce').fillna(0)
        else:
            return jsonify(success=False, message='交易记录缺少total_balance列')
        
        if len(bal) < 2:
            return jsonify(success=False, message='交易记录数据不足')
        
        # 计算收益率
        rets = bal.pct_change().dropna()
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(rets) == 0:
            return jsonify(success=False, message='无法计算收益率')
        
        # 计算最大回撤
        rolling_max = bal.expanding().max()
        drawdown = (bal - rolling_max) / rolling_max
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # 计算夏普比率
        if rets.std() > 0:
            sharpe = float((rets.mean() / rets.std()) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # 计算胜率：盈利的卖出次数 / 总卖出次数
        if 'operate' in df.columns and 'investment' in df.columns:
            sell_mask = df['operate'] == 'sell'
            sell_trades = df[sell_mask]
            if len(sell_trades) > 0:
                profitable_trades = len(sell_trades[pd.to_numeric(sell_trades['investment'], errors='coerce') > 0])
                win_rate = (profitable_trades / len(sell_trades)) * 100
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0
        
        # 计算波动率
        vol = float(rets.std() * np.sqrt(252) * 100) if len(rets) > 1 else 0.0
        
        result = {
            'max_drawdown': abs(max_dd * 100),
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'volatility': vol
        }
        
        print(f"风控指标计算结果: {result}")
        return jsonify(success=True, risk_metrics=result)
        
    except Exception as e:
        print(f"风控计算异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'风险计算失败: {e}')

@app.route('/api/position/calculate', methods=['POST'])
@login_required
def api_position_calculate():
    try:
        d = request.get_json() or {}
        total_capital = d['total_capital']
        method = d.get('method', 'fixed_fractional')
        ps = PositionSizer(total_capital)
        if method == 'fixed_fractional':
            pos = ps.fixed_fractional(d.get('risk_per_trade', 0.02))
        elif method == 'kelly':
            k = ps.kelly_criterion(d.get('win_rate', 0.6), d.get('avg_win', 0.05), d.get('avg_loss', 0.03))
            pos = total_capital * k
        elif method == 'volatility':
            ticker = d['ticker']
            p = os.path.join('data', f'{ticker}.csv')
            if not os.path.isfile(p):
                return jsonify(success=False, message='无法获取股票数据计算波动率')
            df = pd.read_csv(p)
            vol = df['Close'].pct_change().dropna().std() * np.sqrt(252)
            frac = ps.volatility_based_sizing(vol, d.get('target_volatility', 0.15))
            pos = total_capital * frac
        else:
            return jsonify(success=False, message='不支持的仓位计算方法')
        return jsonify(success=True, position_size=pos, position_percentage=pos/total_capital*100, method=method)
    except Exception as e:
        return jsonify(success=False, message=f'仓位计算失败: {e}')

@app.route('/api/stop_loss/set', methods=['POST'])
@login_required
def api_stop_loss_set():
    try:
        d = request.get_json() or {}
        order = stop_m.set_stop_loss(d['ticker'].upper(), d['entry_price'], d.get('stop_loss_pct', 0.05), d.get('trailing', False))
        return jsonify(success=True, order=order)
    except Exception as e:
        return jsonify(success=False, message=f'设置止损失败: {e}')

@app.route('/api/stop_loss/check/<ticker>/<float:current_price>')
@login_required
def api_stop_loss_check(ticker, current_price):
    try:
        triggered = stop_m.check_stop_loss_trigger(ticker.upper(), current_price)
        return jsonify(success=True, triggered=triggered, order=stop_m.stop_loss_orders.get(ticker.upper()))
    except Exception as e:
        return jsonify(success=False, message=f'检查止损失败: {e}')

# ---------- 数据更新入口（手动触发，可后续挂APScheduler） ----------
@app.route('/api/data/update', methods=['POST'])
def api_data_update():
    try:
        d = request.get_json() or {}
        tickers = d.get('tickers', [])
        if not tickers:
            return jsonify(success=False, message='请提供股票代码列表')
        updater = DataUpdater()
        async def run_update():
            return await updater.update_all_stocks(tickers)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_update())
        loop.close()
        ok = sum(1 for v in results.values() if v)
        return jsonify(success=True, message=f'数据更新完成: {ok}/{len(tickers)} 成功', results=results)
    except Exception as e:
        return jsonify(success=False, message=f'数据更新失败: {e}')

def _ensure_ticker_data(ticker, start, end):
    """返回本地 csv 路径；无数据则返回 None"""
    for base in ['data', 'stock_trading/data']:
        f = os.path.join(base, f'{ticker}.csv')
        if os.path.isfile(f):
            return f
    return None

# ---------- 添加股票（预拉取数据） ----------
@app.route('/api/add_stock', methods=['POST'])
def add_stock():
    """
    添加股票：自动识别A股/美股，A股则通过AKShare拉取最近一年数据保存到 data/
    """
    try:
        d = request.get_json() or {}
        ticker = (d.get('ticker') or '').strip().upper()
        if not ticker:
            return jsonify(success=False, message='请输入股票代码')

        data_dir = os.path.join(BASE_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)

        if validate_stock_code(ticker):
            # A股：拉取最近一年数据
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            try:
                csv_path = download_a_share(ticker, start_date, end_date, out_dir=data_dir)
                if csv_path and os.path.exists(csv_path):
                    try:
                        process_a_share_data(csv_path)  # 覆盖原文件为增强版
                    except Exception as e:
                        print(f'添加股票 {ticker} 特征工程失败: {e}，保留原始数据')
                    row_count = len(pd.read_csv(csv_path))
                    return jsonify(success=True, message=f'A股 {ticker} 数据已添加（{row_count} 行），可开始分析')
                return jsonify(success=False, message=f'获取 {ticker} 数据失败')
            except Exception as e:
                return jsonify(success=False, message=f'AKShare 获取失败: {str(e)}')
        else:
            # 美股：使用 yfinance 拉取最近一年数据
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            try:
                csv_path = download_us_stock(ticker, start_date, end_date, out_dir=data_dir)
                if csv_path and os.path.exists(csv_path):
                    try:
                        process_a_share_data(csv_path)
                    except Exception as e:
                        print(f'添加美股 {ticker} 特征工程失败: {e}，保留原始数据')
                    row_count = len(pd.read_csv(csv_path))
                    return jsonify(success=True, message=f'美股 {ticker} 数据已添加（{row_count} 行），可开始分析')
                return jsonify(success=False, message=f'获取美股 {ticker} 数据失败')
            except Exception as e:
                return jsonify(success=False, message=f'yfinance 获取失败: {str(e)}')

    except Exception as e:
        return jsonify(success=False, message=f'添加股票失败: {str(e)}')

# ---------- 高级功能API ----------
# 导出PDF报告
@app.route('/api/report/export', methods=['GET', 'POST'])
def export_report():
    try:
        if request.method == 'POST':
            d = request.get_json() or {}
            ticker = (d.get('ticker') or 'AAPL').upper()
        else:
            ticker = (request.args.get('ticker') or 'AAPL').upper()

        # 读取指标（若存在特定输出文件则尽量使用）
        metrics = {}
        try:
            csv_path = os.path.join(SAVE_DIR, 'output', f'{ticker}_prediction_metrics.csv')
            if os.path.isfile(csv_path):
                dfm = pd.read_csv(csv_path)
                if not dfm.empty:
                    row = dfm.iloc[0].to_dict()
                    metrics = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        except Exception:
            pass

        charts = {
            'prediction': os.path.join(SAVE_DIR, 'pic', 'predictions', f'{ticker}_prediction.png'),
            'loss': os.path.join(SAVE_DIR, 'pic', 'loss', f'{ticker}_loss.png'),
            'earnings': os.path.join(SAVE_DIR, 'pic', 'earnings', f'{ticker}_cumulative.png'),
            'trades': os.path.join(SAVE_DIR, 'pic', 'trades', f'{ticker}_trades.png')
        }

        analysis_data = {'ticker': ticker, 'metrics': metrics}

        pdf_path = generate_analysis_report(ticker, analysis_data, charts, metrics)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return jsonify(success=False, message=f'报告导出失败：{e}')

# 投资组合优化
@app.route('/api/advanced/portfolio/optimize', methods=['POST'])
@login_required
def portfolio_optimize():
    try:
        data = request.get_json() or {}
        method = data.get('method', 'mean_variance')
        tickers = data.get('tickers', [])
        price_data = data.get('price_data')  # 可选，如果提供则直接使用
        
        # 如果没有提供price_data，则从tickers读取本地数据
        if not price_data:
            if not tickers:
                return jsonify(success=False, message='请提供股票代码列表(tickers)或价格数据(price_data)')
            
            # 从本地data目录读取多个股票数据
            price_dict = {}
            for ticker in tickers:
                ticker = str(ticker).strip().upper()
                src = None
                for base in ['data', 'stock_trading/data']:
                    p = os.path.join(base, f'{ticker}.csv')
                    if os.path.isfile(p):
                        src = p
                        break
                
                if not src:
                    return jsonify(success=False, message=f'找不到 {ticker} 的数据文件，请确保data目录下有对应的CSV文件')
                
                df = pd.read_csv(src)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.set_index('Date').sort_index()
                
                if 'Close' in df.columns:
                    price_dict[ticker] = df['Close']
                else:
                    return jsonify(success=False, message=f'{ticker} 数据缺少Close列')
            
            # 合并为DataFrame，对齐日期
            price_df = pd.DataFrame(price_dict)
            price_df = price_df.dropna()  # 删除缺失值
            
            if len(price_df) < 20:
                return jsonify(success=False, message='数据量不足，至少需要20个交易日的数据')
        else:
            # 使用提供的price_data
            price_df = pd.DataFrame(price_data)
            if 'Date' in price_df.columns:
                price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
                price_df = price_df.set_index('Date').sort_index()
        
        # 准备数据
        returns = portfolio_optimizer.prepare_data(price_df)
        
        # 执行优化
        target_return = data.get('target_return')
        if method == 'mean_variance':
            result = portfolio_optimizer.mean_variance_optimization(target_return=target_return)
        elif method == 'risk_parity':
            result = portfolio_optimizer.risk_parity_optimization()
        elif method == 'minimum_variance':
            result = portfolio_optimizer.minimum_variance_optimization()
        else:
            return jsonify(success=False, message='不支持的优化方法')
        
        return jsonify(success=True, result=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'投资组合优化失败: {str(e)}')

# 多时间框架分析
@app.route('/api/advanced/timeframe/analyze', methods=['POST'])
@login_required
def timeframe_analyze():
    try:
        data = request.get_json() or {}
        ticker = data.get('ticker', '').strip().upper()
        price_data = data.get('price_data')  # 可选
        primary_timeframe = data.get('timeframe', '1d')
        timeframes = data.get('timeframes', ['1D', '1W', '1M', '3M'])
        
        # 如果没有提供price_data，则从ticker读取本地数据
        if not price_data:
            if not ticker:
                return jsonify(success=False, message='请提供股票代码(ticker)或价格数据(price_data)')
            
            # 从本地data目录读取数据
            src = None
            for base in ['data', 'stock_trading/data']:
                p = os.path.join(base, f'{ticker}.csv')
                if os.path.isfile(p):
                    src = p
                    break
            
            if not src:
                return jsonify(success=False, message=f'找不到 {ticker} 的数据文件，请确保data目录下有对应的CSV文件')
            
            df = pd.read_csv(src)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date').sort_index()
            else:
                return jsonify(success=False, message='数据文件缺少Date列')
            
            # 确保有必要的列（多时间框架分析需要OHLCV）
            required_cols = ['Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify(success=False, message=f'数据文件缺少必要的列: {", ".join(missing_cols)}')
            
            # 如果缺少OHLCV中的某些列，尝试用Close填充
            if 'Open' not in df.columns:
                df['Open'] = df['Close']
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            if 'Volume' not in df.columns:
                df['Volume'] = 0  # 如果没有成交量数据，设为0
        else:
            # 使用提供的price_data
            if isinstance(price_data, dict):
                df = pd.DataFrame(price_data)
            elif isinstance(price_data, list):
                df = pd.DataFrame(price_data)
            else:
                df = pd.DataFrame(price_data)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date').sort_index()
            elif df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index, errors='coerce')
            
            # 确保有必要的列
            if 'Close' not in df.columns:
                return jsonify(success=False, message='价格数据缺少Close列')
            
            # 如果缺少OHLCV中的某些列，尝试用Close填充
            if 'Open' not in df.columns:
                df['Open'] = df['Close']
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            if 'Volume' not in df.columns:
                df['Volume'] = 0
        
        df = df.dropna()
        
        if len(df) < 20:
            return jsonify(success=False, message='数据量不足，至少需要20个交易日的数据')
        
        # 执行多时间框架分析
        results = timeframe_analyzer.multi_timeframe_analysis(df, primary_timeframe)
        
        return jsonify(success=True, results=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'多时间框架分析失败: {str(e)}')

# 机器学习流水线
@app.route('/api/advanced/ml/pipeline', methods=['POST'])
@login_required
def ml_pipeline_analysis():
    try:
        data = request.get_json() or {}
        ticker = data.get('ticker', '').strip().upper()
        price_data = data.get('price_data')  # 可选
        target_col = data.get('target_col', 'future_return_1')
        models = data.get('models', ['RandomForest', 'XGBoost', 'LightGBM'])
        
        # 如果没有提供price_data，则从ticker读取本地数据
        if not price_data:
            if not ticker:
                return jsonify(success=False, message='请提供股票代码(ticker)或价格数据(price_data)')
            
            # 从本地data目录读取数据
            src = None
            for base in ['data', 'stock_trading/data']:
                p = os.path.join(base, f'{ticker}.csv')
                if os.path.isfile(p):
                    src = p
                    break
            
            if not src:
                return jsonify(success=False, message=f'找不到 {ticker} 的数据文件，请确保data目录下有对应的CSV文件')
            
            df = pd.read_csv(src)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date').sort_index()
            else:
                return jsonify(success=False, message='数据文件缺少Date列')
            
            # 确保有必要的列
            if 'Close' not in df.columns:
                return jsonify(success=False, message='数据文件缺少Close列')
        else:
            # 使用提供的price_data
            df = pd.DataFrame(price_data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date').sort_index()
            elif df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index, errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 50:
            return jsonify(success=False, message='数据量不足，至少需要50个交易日的数据')
        
        # 执行机器学习流水线
        results = ml_pipeline.full_pipeline(df, target_col)
        
        return jsonify(success=True, results=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'机器学习流水线失败: {str(e)}')

# 机构级风险管理
@app.route('/api/advanced/risk/institutional', methods=['POST'])
@login_required
def institutional_risk_analysis():
    try:
        data = request.get_json()
        portfolio = data.get('portfolio', {})
        returns_data = data.get('returns_data', [])
        
        if not portfolio or not returns_data:
            return jsonify(success=False, message='请提供投资组合和收益率数据')
        
        # 准备数据
        returns_df = pd.DataFrame(returns_data)
        
        # 执行机构级风险分析
        report = institutional_risk.comprehensive_risk_report(
            portfolio, returns_df, {}
        )
        
        return jsonify(success=True, report=report)
    except Exception as e:
        return jsonify(success=False, message=str(e))

# 合规检查
@app.route('/api/advanced/compliance/check', methods=['POST'])
@login_required
def compliance_check():
    try:
        data = request.get_json()
        portfolio_data = data.get('portfolio_data', {})
        
        if not portfolio_data:
            return jsonify(success=False, message='请提供投资组合数据')
        
        # 执行合规检查
        results = compliance.comprehensive_compliance_check(portfolio_data)
        
        return jsonify(success=True, results=results)
    except Exception as e:
        return jsonify(success=False, message=str(e))

# 白标配置
@app.route('/api/advanced/whitelabel/config', methods=['GET', 'POST'])
@login_required
def whitelabel_config():
    if request.method == 'GET':
        # 获取当前白标配置
        tenant_id = session.get('tenant_id', 'default')
        config = white_label.generate_config_file(tenant_id)
        return jsonify(success=True, config=config)
    
    elif request.method == 'POST':
        # 更新白标配置
        data = request.get_json()
        tenant_id = session.get('tenant_id', 'default')
        
        if 'branding' in data:
            white_label.update_branding(tenant_id, data['branding'])
        if 'theme' in data:
            white_label.update_theme(tenant_id, data['theme'])
        if 'features' in data:
            white_label.update_features(tenant_id, data['features'])
        
        return jsonify(success=True, message='配置更新成功')

# ---------- 用户中心 ----------
@app.route('/api/user/profile')
@login_required
def user_profile(): 
    if not session.get('user_id'):
        return jsonify(success=False, message='请先登录'), 401
    return jsonify(db['user_manager'].get_user_profile(session['user_id']))


@app.route('/api/user/profile', methods=['PUT'])
@login_required
def user_profile_update():
    """更新当前用户资料（昵称/邮箱/头像/投资偏好）"""
    try:
        if not session.get('user_id'):
            return jsonify(success=False, message='请先登录'), 401
        user_id = int(session.get('user_id'))
        d = request.get_json() or {}
        username = (d.get('username') or '').strip()
        email = (d.get('email') or '').strip()
        avatar_url = (d.get('avatar_url') or '').strip()
        investment_preference = (d.get('investment_preference') or '').strip()
        risk_tolerance = (d.get('risk_tolerance') or '').strip()
        investment_goal = (d.get('investment_goal') or '').strip()

        conn = db['db'].get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
            old_row = cursor.fetchone()
            if not old_row:
                return jsonify(success=False, message='用户不存在')

            if username and username != old_row['username']:
                if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
                    return jsonify(success=False, message='用户名只能包含3-20位字母、数字或下划线')
                cursor.execute('SELECT id FROM users WHERE username = ? AND id <> ?', (username, user_id))
                if cursor.fetchone():
                    return jsonify(success=False, message='用户名已存在')
            else:
                username = old_row['username']

            if email:
                if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                    return jsonify(success=False, message='邮箱格式不正确')
                cursor.execute('SELECT id FROM users WHERE email = ? AND id <> ?', (email, user_id))
                if cursor.fetchone():
                    return jsonify(success=False, message='邮箱已被占用')
            else:
                email = old_row['email']

            cursor.execute('''
                UPDATE users
                SET username = ?, email = ?, avatar_url = ?,
                    investment_preference = ?, risk_tolerance = ?, investment_goal = ?
                WHERE id = ?
            ''', (
                username, email, avatar_url or None,
                investment_preference or None, risk_tolerance or None, investment_goal or None,
                user_id
            ))
            cursor.execute('''
                INSERT INTO audit_logs (user_id, action, details, ip_address, timestamp)
                VALUES (?, 'USER_PROFILE_UPDATE', ?, ?, ?)
            ''', (user_id, f'username={username}, email={email}', request.remote_addr, datetime.now()))
            conn.commit()
        finally:
            conn.close()

        session['username'] = username
        return jsonify(success=True, message='个人资料已更新')
    except Exception as e:
        return jsonify(success=False, message=f'更新失败: {e}')


def _is_allowed_avatar_file(filename: str) -> bool:
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in ALLOWED_AVATAR_EXTS


@app.route('/api/user/avatar', methods=['POST'])
@login_required
def user_avatar_upload():
    """头像本地上传"""
    try:
        if not session.get('user_id'):
            return jsonify(success=False, message='请先登录'), 401
        if 'avatar' not in request.files:
            return jsonify(success=False, message='未检测到上传文件')
        f = request.files.get('avatar')
        if not f or not f.filename:
            return jsonify(success=False, message='请选择头像文件')
        if not _is_allowed_avatar_file(f.filename):
            return jsonify(success=False, message='仅支持 jpg/jpeg/png/webp 格式')

        # 文件大小限制 2MB
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        if size > MAX_AVATAR_SIZE:
            return jsonify(success=False, message='头像不能超过 2MB')

        user_id = int(session.get('user_id'))
        ext = secure_filename(f.filename).rsplit('.', 1)[-1].lower()
        new_name = f'u{user_id}_{int(time.time())}_{os.urandom(4).hex()}.{ext}'
        save_path = os.path.join(AVATAR_UPLOAD_DIR, new_name)
        f.save(save_path)
        avatar_url = f'/uploads/avatars/{new_name}'

        conn = db['db'].get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT avatar_url FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            old_avatar = (row['avatar_url'] if row and 'avatar_url' in row.keys() else None) or ''
            cursor.execute('UPDATE users SET avatar_url = ? WHERE id = ?', (avatar_url, user_id))
            conn.commit()
        finally:
            conn.close()

        # 清理旧头像（只删本地上传目录的文件）
        if old_avatar and old_avatar.startswith('/uploads/avatars/'):
            old_name = old_avatar.split('/uploads/avatars/', 1)[-1]
            old_path = os.path.join(AVATAR_UPLOAD_DIR, old_name)
            if os.path.isfile(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass

        return jsonify(success=True, message='头像上传成功', avatar_url=avatar_url)
    except Exception as e:
        return jsonify(success=False, message=f'头像上传失败: {e}')

@app.route('/api/analysis/history')
@login_required
def analysis_history():
    return jsonify(success=True, analyses=db['analysis_manager'].get_user_analyses(session['user_id'], 50))

@app.route('/api/favorites')
@login_required
def favorites():
    return jsonify(success=True, favorites=db['favorite_manager'].get_user_favorites(session['user_id']))

@app.route('/api/favorites/<ticker>', methods=['POST'])
@login_required
def add_fav(ticker):
    suc = db['favorite_manager'].add_favorite(session['user_id'], ticker.upper(), request.json.get('notes', ''))
    return jsonify(success=suc, message='已收藏' if suc else '失败')

@app.route('/api/favorites/<ticker>', methods=['DELETE'])
@login_required
def del_fav(ticker):
    suc = db['favorite_manager'].remove_favorite(session['user_id'], ticker.upper())
    return jsonify(success=suc, message='已移除' if suc else '失败')

# ---------- 静态图片 ----------
@app.route('/uploads/avatars/<path:filename>')
def serve_uploaded_avatar(filename):
    return send_from_directory(AVATAR_UPLOAD_DIR, filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    # 兼容旧路径：/images/predictions/xxx.png 或直接 /images/xxx.png
    p = os.path.join(SAVE_DIR, 'pic', filename)
    if os.path.isfile(p):
        return send_file(p)
    # 若传入不含子目录，优先从 predictions 查找
    pred = os.path.join(SAVE_DIR, 'pic', 'predictions', filename)
    if os.path.isfile(pred):
        return send_file(pred)
    return jsonify(success=False, message='图片不存在'), 404

# 递归暴露所有子目录
for _sub in ['predictions', 'loss', 'earnings', 'trades']:
    subdir = os.path.join(SAVE_DIR, 'pic', _sub)
    os.makedirs(subdir, exist_ok=True)
    app.add_url_rule(f'/images/{_sub}/<path:filename>', f'serve_image_{_sub}',
                     (lambda sd=subdir: (lambda filename: send_from_directory(sd, filename)))())

# 供前端原始数据/交易记录弹窗读取 CSV
@app.route('/tmp/flask/ticker/<path:filename>')
def serve_tmp_ticker_csv(filename):
    # 为临时 ticker 目录暴露静态路由
    ticker_dir = os.path.join(TEMP_DIR, 'ticker')
    os.makedirs(ticker_dir, exist_ok=True)
    return send_from_directory(ticker_dir, filename)

@app.route('/results/transactions/<path:filename>')
def serve_results_transactions_csv(filename):
    p = os.path.join(SAVE_DIR, 'transactions', filename)
    if not os.path.isfile(p):
        return jsonify(success=False, message='文件不存在'), 404
    resp = send_file(p, as_attachment=False)
    # 设置1分钟缓存，允许条件验证
    resp.headers['Cache-Control'] = 'max-age=60, must-revalidate'
    return resp

# 添加数据目录的静态路由
@app.route('/data/<path:filename>')
def serve_data_csv(filename):
    """提供data目录下的CSV文件"""
    data_dir = 'data'
    if os.path.isdir(data_dir):
        return send_from_directory(data_dir, filename)
    return jsonify(success=False, message='数据目录不存在'), 404

@app.route('/stock_trading/data/<path:filename>')
def serve_stock_trading_data_csv(filename):
    """提供stock_trading/data目录下的CSV文件"""
    data_dir = 'stock_trading/data'
    if os.path.isdir(data_dir):
        return send_from_directory(data_dir, filename)
    return jsonify(success=False, message='股票交易数据目录不存在'), 404

# ---------- 分析结果预览页面 ----------
@app.route('/analysis/view')
def analysis_view():
     try:
         ticker = (request.args.get('ticker') or '').upper()
         if not ticker:
             return jsonify(success=False, message='缺少股票代码ticker')
         images = {
             'prediction': f"/images/predictions/{ticker}_prediction.png",
             'loss': f"/images/loss/{ticker}_loss.png",
             'earnings': f"/images/earnings/{ticker}_cumulative.png",
             'trades': f"/images/trades/{ticker}_trades.png"
         }
         return render_template('analysis_view.html', ticker=ticker, images=images)
     except Exception as e:
         return jsonify(success=False, message=str(e))

# ---------- 增强版分析报告导出 ----------
@app.route('/api/analysis/export_comprehensive', methods=['POST'])
@login_required
def export_comprehensive_analysis():
    """导出完整的分析报告（包含所有图表和指标）到下载目录"""
    try:
        d = request.get_json() or {}
        ticker = (d.get('ticker') or 'AAPL').upper()
        format_type = d.get('format', 'pdf')  # pdf 或 word
        
        # 获取用户下载目录
        import os
        from pathlib import Path
        
        # 尝试获取用户下载目录
        downloads_dir = None
        try:
            # Windows
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            if not os.path.exists(downloads_dir):
                # 备用方案
                downloads_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        except:
            downloads_dir = os.path.join(BASE_DIR, "downloads")
        
        os.makedirs(downloads_dir, exist_ok=True)
        
        # 收集所有分析数据
        analysis_data = {
            'ticker': ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
            'trading_results': {},
            'transactions': [],
            'charts': {}
        }
        
        # 1. 读取预测指标
        try:
            metrics_file = os.path.join(SAVE_DIR, 'output', f'{ticker}_prediction_metrics.csv')
            if os.path.isfile(metrics_file):
                df_metrics = pd.read_csv(metrics_file)
                if not df_metrics.empty:
                    analysis_data['metrics'] = df_metrics.iloc[0].to_dict()
        except Exception as e:
            print(f"读取预测指标失败: {e}")
        
        # 2. 读取交易结果
        try:
            trading_file = os.path.join(SAVE_DIR, 'output', f'{ticker}_trading_metrics.csv')
            if os.path.isfile(trading_file):
                df_trading = pd.read_csv(trading_file)
                if not df_trading.empty:
                    analysis_data['trading_results'] = df_trading.iloc[0].to_dict()
        except Exception as e:
            print(f"读取交易结果失败: {e}")
        
        # 3. 读取交易记录
        try:
            trans_file = os.path.join(SAVE_DIR, 'transactions', f'{ticker}_transactions.csv')
            if os.path.isfile(trans_file):
                df_trans = pd.read_csv(trans_file)
                analysis_data['transactions'] = df_trans.to_dict('records')
        except Exception as e:
            print(f"读取交易记录失败: {e}")
        
        # 4. 收集图表路径
        chart_paths = {
            'prediction': os.path.join(SAVE_DIR, 'pic', 'predictions', f'{ticker}_prediction.png'),
            'loss': os.path.join(SAVE_DIR, 'pic', 'loss', f'{ticker}_loss.png'),
            'earnings': os.path.join(SAVE_DIR, 'pic', 'earnings', f'{ticker}_cumulative.png'),
            'trades': os.path.join(SAVE_DIR, 'pic', 'trades', f'{ticker}_trades.png')
        }
        
        # 检查图表文件是否存在
        for chart_name, chart_path in chart_paths.items():
            if os.path.isfile(chart_path):
                analysis_data['charts'][chart_name] = chart_path
        
        # 5. 生成报告
        if format_type.lower() == 'pdf':
            # 使用现有的PDF生成器
            pdf_path = generate_analysis_report(ticker, analysis_data, chart_paths, analysis_data['metrics'])
            
            # 复制到下载目录
            import shutil
            final_path = os.path.join(downloads_dir, f'{ticker}_分析报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            shutil.copy2(pdf_path, final_path)
            
            return jsonify(success=True, message=f'PDF报告已保存到: {final_path}', file_path=final_path)
            
        elif format_type.lower() == 'word':
            # 生成Word报告
            try:
                import importlib

                docx_module = importlib.import_module('docx')
                shared_module = importlib.import_module('docx.shared')
                enum_module = importlib.import_module('docx.enum.text')

                Document = getattr(docx_module, 'Document')
                Inches = getattr(shared_module, 'Inches')
                WD_ALIGN_PARAGRAPH = getattr(enum_module, 'WD_ALIGN_PARAGRAPH')
            except (ImportError, AttributeError):
                return jsonify(success=False, message='需要安装python-docx库: pip install python-docx')

            try:
                doc = Document()
                
                # 标题
                title = doc.add_heading(f'{ticker} 股票分析报告', 0)
                title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # 基本信息
                doc.add_heading('基本信息', level=1)
                info_table = doc.add_table(rows=3, cols=2)
                info_table.style = 'Table Grid'
                info_table.cell(0, 0).text = '股票代码'
                info_table.cell(0, 1).text = ticker
                info_table.cell(1, 0).text = '分析时间'
                info_table.cell(1, 1).text = analysis_data['timestamp']
                info_table.cell(2, 0).text = '报告类型'
                info_table.cell(2, 1).text = 'LSTM预测 + 强化学习交易'
                
                # 预测指标
                if analysis_data['metrics']:
                    doc.add_heading('预测指标', level=1)
                    metrics_table = doc.add_table(rows=len(analysis_data['metrics'])+1, cols=2)
                    metrics_table.style = 'Table Grid'
                    metrics_table.cell(0, 0).text = '指标名称'
                    metrics_table.cell(0, 1).text = '数值'
                    
                    row_idx = 1
                    for key, value in analysis_data['metrics'].items():
                        metrics_table.cell(row_idx, 0).text = str(key)
                        metrics_table.cell(row_idx, 1).text = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
                        row_idx += 1
                
                # 交易结果
                if analysis_data['trading_results']:
                    doc.add_heading('交易结果', level=1)
                    trading_table = doc.add_table(rows=len(analysis_data['trading_results'])+1, cols=2)
                    trading_table.style = 'Table Grid'
                    trading_table.cell(0, 0).text = '指标名称'
                    trading_table.cell(0, 1).text = '数值'
                    
                    row_idx = 1
                    for key, value in analysis_data['trading_results'].items():
                        trading_table.cell(row_idx, 0).text = str(key)
                        trading_table.cell(row_idx, 1).text = f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)
                        row_idx += 1
                
                # 添加图表
                doc.add_heading('分析图表', level=1)
                
                for chart_name, chart_path in analysis_data['charts'].items():
                    if os.path.isfile(chart_path):
                        doc.add_heading(f'{chart_name} 图表', level=2)
                        doc.add_picture(chart_path, width=Inches(6))
                
                # 交易记录摘要
                if analysis_data['transactions']:
                    doc.add_heading('交易记录摘要', level=1)
                    trans_df = pd.DataFrame(analysis_data['transactions'])
                    
                    # 统计信息
                    total_trades = len(trans_df)
                    buy_trades = len(trans_df[trans_df['operate'] == 'buy'])
                    sell_trades = len(trans_df[trans_df['operate'] == 'sell'])
                    
                    doc.add_paragraph(f'总交易次数: {total_trades}')
                    doc.add_paragraph(f'买入次数: {buy_trades}')
                    doc.add_paragraph(f'卖出次数: {sell_trades}')
                    
                    # 最近10笔交易
                    doc.add_heading('最近10笔交易', level=2)
                    recent_trans = trans_df.tail(10)
                    
                    trans_table = doc.add_table(rows=len(recent_trans)+1, cols=len(recent_trans.columns))
                    trans_table.style = 'Table Grid'
                    
                    # 表头
                    for i, col in enumerate(recent_trans.columns):
                        trans_table.cell(0, i).text = str(col)
                    
                    # 数据行
                    for i, (_, row) in enumerate(recent_trans.iterrows(), 1):
                        for j, value in enumerate(row):
                            trans_table.cell(i, j).text = str(value)
                
                # 保存Word文档
                word_filename = f'{ticker}_分析报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx'
                word_path = os.path.join(downloads_dir, word_filename)
                doc.save(word_path)
                
                return jsonify(success=True, message=f'Word报告已保存到: {word_path}', file_path=word_path)
                
            except Exception as e:
                return jsonify(success=False, message=f'生成Word报告失败: {e}')
        
        else:
            return jsonify(success=False, message='不支持的格式，请选择pdf或word')
            
    except Exception as e:
        print(f"导出综合报告失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, message=f'导出失败: {e}')

# ---------- 启动应用 ----------
if __name__ == '__main__':
    print("智能股票交易系统（增强版）启动中...")
    print("功能模块:")
    print("  - 用户认证系统")
    print("  - 股票数据获取")
    print("  - LSTM预测分析")
    print("  - 强化学习交易")
    print("  - 投资组合优化")
    print("  - 多时间框架分析")
    print("  - 机器学习流水线")
    print("  - 机构级风险管理")
    print("  - 合规检查工具")
    print("  - 白标解决方案")
    print("访问地址: http://127.0.0.1:5000")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"启动失败: {e}")
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
