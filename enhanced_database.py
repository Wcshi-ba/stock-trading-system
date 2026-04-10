#!/usr/bin/env python3
"""
增强数据库模块 - 为客户提供更好的用户体验
包含用户管理、交易记录、分析历史、收藏夹等功能
"""

import sqlite3
import hashlib
import bcrypt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import os

class EnhancedDatabase:
    """增强数据库类"""
    
    def __init__(self, db_path: str = "enhanced_trading_system.db"):
        # 统一使用绝对路径，避免不同工作目录下生成多个数据库文件
        if not os.path.isabs(db_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, db_path)
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        return conn
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                preferences TEXT,
                avatar_url VARCHAR(255),
                investment_preference TEXT,
                risk_tolerance VARCHAR(20) DEFAULT 'medium',
                investment_goal VARCHAR(50) DEFAULT 'growth',
                role VARCHAR(20) DEFAULT 'user'
            )
        ''')
        
        # 用户会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 分析历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                analysis_type VARCHAR(50) NOT NULL,
                parameters TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time REAL,
                accuracy REAL,
                total_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                is_favorite BOOLEAN DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                analysis_id INTEGER NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                action VARCHAR(10) NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                profit_loss REAL,
                total_balance REAL,
                strategy_used VARCHAR(50),
                confidence_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
            )
        ''')
        
        # 用户收藏表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                UNIQUE(user_id, ticker),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 系统配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                key VARCHAR(100) PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 角色权限表（管理员可配置各角色权限）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS role_permissions (
                role_name VARCHAR(30) NOT NULL,
                permission_key VARCHAR(80) NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (role_name, permission_key)
            )
        ''')
        
        # 用户反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                feedback_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                admin_reply TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 投资组合表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                stock_symbol VARCHAR(10) NOT NULL,
                shares INTEGER NOT NULL DEFAULT 0,
                avg_price DECIMAL(10,4) NOT NULL DEFAULT 0,
                current_price DECIMAL(10,4) DEFAULT 0,
                total_value DECIMAL(12,2) DEFAULT 0,
                unrealized_pnl DECIMAL(12,2) DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, stock_symbol)
            )
        ''')
        
        # 用户交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                stock_symbol VARCHAR(10) NOT NULL,
                action VARCHAR(10) NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                quantity INTEGER NOT NULL,
                total_amount DECIMAL(12,2) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_used VARCHAR(50),
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 用户设置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                setting_key VARCHAR(50) NOT NULL,
                setting_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, setting_key)
            )
        ''')
        
        # 数据缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key VARCHAR(255) UNIQUE NOT NULL,
                cache_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                cache_type VARCHAR(50)
            )
        ''')
        
        # 审计日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action VARCHAR(50) NOT NULL,
                details TEXT,
                ip_address VARCHAR(45),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 密码重置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                reset_code VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_used BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # 管理员任务监控表（可追踪执行状态/进度）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type VARCHAR(50) NOT NULL,
                task_name VARCHAR(120),
                user_id INTEGER,
                ticker VARCHAR(20),
                status VARCHAR(20) DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                detail TEXT,
                source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # 系统指标历史快照（支持系统监控曲线）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                process_memory_mb REAL,
                active_users INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 常用查询索引（提升管理员页列表与统计性能）
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_history_user_created ON analysis_history(user_id, created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_history_created ON analysis_history(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_records_user_time ON trading_records(user_id, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_user_created ON user_feedback(user_id, created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_status_created ON user_feedback(status, created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_runs_status_created ON admin_task_runs(status, created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_created ON system_metrics_snapshots(created_at)')
        
        conn.commit()
        conn.close()
        print("数据库初始化完成")

    def get_config(self, key: str, default: str = "") -> str:
        """获取系统配置"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row['value'] if row else default
        finally:
            conn.close()

    def set_config(self, key: str, value: str) -> None:
        """设置系统配置"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
            conn.commit()
        finally:
            conn.close()


class RolePermissionManager:
    """角色权限管理 - 从数据库读取/更新权限"""
    
    # 权限键与中文名称映射（供管理界面展示）
    PERMISSION_LABELS = {
        'strategy_management': '策略使用管理',
        'model_config': '模型与策略配置',
        'model_multi_integration': '多模型集成',
        'backtest_strategies': '回测策略',
        'risk_control': '风控规则',
        'data_export': '数据导出',
        'market_data.view': '查看行情数据',
        'market_data.export_basic': '导出基础数据',
        'strategy.simulate_trade': '模拟交易',
        'strategy.view_backtest': '查看回测',
        'model.view_predictions': '查看预测',
        'risk.view_personal': '查看个人风控',
        'risk.set_personal_threshold': '设置个人风控',
        'account.modify_own': '修改个人资料',
        'account.view_all_logs': '查看全部日志',
        'account.assign_role': '分配角色',
        'account.config_quota': '配置配额',
    }
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
        self._seed_if_empty()
    
    def _seed_if_empty(self):
        """若 role_permissions 为空，从 permissions.PERMISSION_MATRIX 初始化"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM role_permissions")
            if cursor.fetchone()[0] > 0:
                return
            from permissions import PermissionManager
            # 扩展权限：策略管理、模型配置等（admin 有，user 默认无）
            extra_admin = {'strategy_management', 'model_config', 'model_multi_integration', 'backtest_strategies', 'risk_control', 'data_export'}
            extra_user = {'strategy_management', 'model_config'}  # user 可配置开启
            for role, perms in PermissionManager.PERMISSION_MATRIX.items():
                for perm, enabled in perms.items():
                    if enabled:
                        cursor.execute('''
                            INSERT OR IGNORE INTO role_permissions (role_name, permission_key, enabled)
                            VALUES (?, ?, 1)
                        ''', (role, perm))
                if role == 'admin':
                    for p in extra_admin:
                        cursor.execute('''
                            INSERT OR IGNORE INTO role_permissions (role_name, permission_key, enabled)
                            VALUES (?, ?, 1)
                        ''', (role, p))
                elif role == 'user':
                    for p in extra_user:
                        cursor.execute('''
                            INSERT OR IGNORE INTO role_permissions (role_name, permission_key, enabled)
                            VALUES (?, ?, 0)
                        ''', (role, p))
            conn.commit()
        except Exception as e:
            print(f"初始化角色权限失败: {e}")
        finally:
            conn.close()
    
    def get_all_roles_permissions(self) -> Dict[str, Dict[str, bool]]:
        """返回 {role: {permission: enabled}}"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT role_name, permission_key, enabled FROM role_permissions")
            result = {}
            for row in cursor.fetchall():
                r, p, e = row[0], row[1], bool(row[2])
                if r not in result:
                    result[r] = {}
                result[r][p] = e
            return result
        finally:
            conn.close()
    
    def get_roles(self) -> List[str]:
        """获取所有角色名"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT role_name FROM role_permissions ORDER BY role_name")
            return [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()
    
    def set_permission(self, role_name: str, permission_key: str, enabled: bool) -> bool:
        """设置单个权限"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO role_permissions (role_name, permission_key, enabled, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (role_name, permission_key, 1 if enabled else 0))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_permission_labels(self) -> Dict[str, str]:
        return dict(self.PERMISSION_LABELS)


class UserManager:
    """用户管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def hash_password(self, password: str) -> str:
        """密码哈希（使用bcrypt）"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """验证密码，兼容旧的SHA-256哈希，命中后自动迁移为bcrypt"""
        if not stored_hash:
            return False
        try:
            # bcrypt 格式以 $2 开头
            if stored_hash.startswith('$2'):
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
            # 兼容旧 sha256
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
        except Exception:
            return False
    
    def register_user(self, username: str, password: str, email: str = "", role: str = "user") -> Dict[str, Any]:
        """用户注册 - 支持指定角色，第一个用户自动为 admin"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # 检查用户名是否存在
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return {'success': False, 'message': '用户名已存在'}
            
            # 第一个用户自动设为管理员
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()[0]
            final_role = 'admin' if user_count == 0 else role
            
            # 创建用户（bcrypt）
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, preferences, role) 
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, json.dumps({}), final_role))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                'success': True,
                'user_id': user_id,
                'message': '注册成功',
                'role': final_role
            }
            
        except Exception as e:
            conn.rollback()
            return {'success': False, 'message': f'注册失败: {str(e)}'}
        finally:
            conn.close()
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """用户登录"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # 获取用户后在应用层验证密码（兼容 bcrypt + 旧 sha256）
            cursor.execute('''
                SELECT id, username, email, password_hash, preferences, avatar_url, is_active 
                FROM users 
                WHERE username = ?
            ''', (username,))
            user = cursor.fetchone()
            if not user:
                return {'success': False, 'message': '用户名或密码错误'}
            if user['is_active'] != 1:
                return {'success': False, 'message': '账户已禁用'}

            stored_hash = user['password_hash']
            ok = self.verify_password(password, stored_hash)
            if not ok:
                return {'success': False, 'message': '用户名或密码错误'}

            # 如果命中旧 sha256 则迁移为 bcrypt
            if not stored_hash.startswith('$2'):
                try:
                    new_hash = self.hash_password(password)
                    cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
                    conn.commit()
                except Exception:
                    pass
            
            # 更新最后登录时间
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user['id'],))
            
            conn.commit()
            
            return {
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'preferences': json.loads(user['preferences'] or '{}'),
                    'avatar_url': user['avatar_url']
                }
            }
            
        except Exception as e:
            return {'success': False, 'message': f'登录失败: {str(e)}'}
        finally:
            conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户信息（返回 dict 便于使用）"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, username, email, password_hash, created_at, last_login, 
                       investment_preference, risk_tolerance, investment_goal, role, avatar_url, is_active
                FROM users WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            if user:
                return dict(user)
            return None
            
        except Exception as e:
            print(f"获取用户失败: {e}")
            return None
        finally:
            conn.close()
    
    def update_last_login(self, user_id: int) -> bool:
        """更新最后登录时间"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"更新登录时间失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        """根据用户ID获取用户信息"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, username, email, password_hash, created_at, last_login, 
                       investment_preference, risk_tolerance, investment_goal, role
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            return user
            
        except Exception as e:
            print(f"获取用户失败: {e}")
            return None
        finally:
            conn.close()

    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """获取用户资料"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT username, email, created_at, last_login, preferences, avatar_url, role
                FROM users WHERE id = ? AND is_active = 1
            ''', (user_id,))
            
            user = cursor.fetchone()
            if not user:
                return {'success': False, 'message': '用户不存在'}
            
            # 获取用户统计信息
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(accuracy) as avg_accuracy,
                    AVG(total_return) as avg_return,
                    MAX(created_at) as last_analysis
                FROM analysis_history 
                WHERE user_id = ?
            ''', (user_id,))
            
            stats = cursor.fetchone()
            
            return {
                'success': True,
                'profile': {
                    'username': user['username'],
                    'email': user['email'],
                    'created_at': user['created_at'],
                    'last_login': user['last_login'],
                    'preferences': json.loads(user['preferences'] or '{}'),
                    'avatar_url': user['avatar_url'],
                    'role': user['role'] if 'role' in user.keys() else 'user',
                    'stats': {
                        'total_analyses': stats['total_analyses'] or 0,
                        'avg_accuracy': stats['avg_accuracy'] or 0,
                        'avg_return': stats['avg_return'] or 0,
                        'last_analysis': stats['last_analysis']
                    }
                }
            }
            
        except Exception as e:
            return {'success': False, 'message': f'获取资料失败: {str(e)}'}
        finally:
            conn.close()
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """更新用户偏好设置"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE users SET preferences = ? WHERE id = ?
            ''', (json.dumps(preferences), user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"更新偏好失败: {e}")
            return False
        finally:
            conn.close()
    
    def update_user_role(self, user_id: int, role: str) -> bool:
        """更新用户角色"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE users SET role = ? WHERE id = ?
            ''', (role, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"更新用户角色失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_count(self) -> int:
        """获取用户总数"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            print(f"获取用户总数失败: {e}")
            return 0
        finally:
            conn.close()
    
    def admin_reset_user_password(self, admin_id: int, target_user_id: int, new_password: str) -> bool:
        """管理员强制重置用户密码"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT role FROM users WHERE id = ?", (admin_id,))
            admin = cursor.fetchone()
            admin_dict = dict(admin) if admin else {}
            if not admin_dict.get('role') == 'admin':
                return False
            
            new_hash = self.hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, target_user_id))
            conn.commit()
            
            cursor.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES (?, 'ADMIN_PASSWORD_RESET', ?, ?)
            ''', (admin_id, f'Reset password for user {target_user_id}', datetime.now()))
            conn.commit()
            
            return True
        except Exception as e:
            print(f"管理员重置密码失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, username, email, created_at, last_login, 
                       is_active, role, investment_preference, risk_tolerance
                FROM users 
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                user = dict(row)
                user['status'] = 'active' if user['is_active'] else 'inactive'
                users.append(user)
            
            return users
        except Exception as e:
            print(f"获取用户列表失败: {e}")
            return []
        finally:
            conn.close()

class AnalysisManager:
    """分析管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def save_analysis(self, user_id: int, ticker: str, analysis_type: str, 
                     parameters: Dict, results: Dict, execution_time: float = 0) -> int:
        """保存分析结果"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # 提取关键指标
            accuracy = results.get('prediction_metrics', {}).get('accuracy', 0)
            total_return = results.get('trading_results', {}).get('investment_return', 0)
            
            cursor.execute('''
                INSERT INTO analysis_history 
                (user_id, ticker, analysis_type, parameters, results, 
                 execution_time, accuracy, total_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, ticker, analysis_type, json.dumps(parameters), 
                  json.dumps(results), execution_time, accuracy, total_return))
            
            analysis_id = cursor.lastrowid
            
            # 保存交易记录
            if 'transactions' in results:
                self.save_trading_records(cursor, user_id, analysis_id, ticker, results['transactions'])
            
            conn.commit()
            return analysis_id
            
        except Exception as e:
            conn.rollback()
            print(f"保存分析失败: {e}")
            return 0
        finally:
            conn.close()
    
    def save_trading_records(self, cursor, user_id: int, analysis_id: int, 
                           ticker: str, transactions: List[Dict]):
        """保存交易记录"""
        for transaction in transactions:
            cursor.execute('''
                INSERT INTO trading_records 
                (user_id, analysis_id, ticker, action, price, 
                 profit_loss, total_balance, strategy_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, analysis_id, ticker, transaction.get('operate'),
                  transaction.get('price'), transaction.get('investment'),
                  transaction.get('total_balance'), 'AI_Strategy'))
    
    def get_user_analyses(self, user_id: int, limit: int = 50) -> List[Dict]:
        """获取用户分析历史"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, ticker, analysis_type, created_at, accuracy, 
                       total_return, max_drawdown, sharpe_ratio, is_favorite, notes
                FROM analysis_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            analyses = []
            for row in cursor.fetchall():
                analyses.append({
                    'id': row['id'],
                    'ticker': row['ticker'],
                    'analysis_type': row['analysis_type'],
                    'created_at': row['created_at'],
                    'accuracy': row['accuracy'],
                    'total_return': row['total_return'],
                    'max_drawdown': row['max_drawdown'],
                    'sharpe_ratio': row['sharpe_ratio'],
                    'is_favorite': row['is_favorite'],
                    'notes': row['notes']
                })
            
            return analyses
            
        except Exception as e:
            print(f"获取分析历史失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_analysis_details(self, analysis_id: int, user_id: int) -> Dict[str, Any]:
        """获取分析详情"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM analysis_history 
                WHERE id = ? AND user_id = ?
            ''', (analysis_id, user_id))
            
            analysis = cursor.fetchone()
            if not analysis:
                return {'success': False, 'message': '分析不存在'}
            
            # 获取交易记录
            cursor.execute('''
                SELECT * FROM trading_records 
                WHERE analysis_id = ? AND user_id = ?
                ORDER BY timestamp
            ''', (analysis_id, user_id))
            
            trades = cursor.fetchall()
            
            return {
                'success': True,
                'analysis': {
                    'id': analysis['id'],
                    'ticker': analysis['ticker'],
                    'analysis_type': analysis['analysis_type'],
                    'parameters': json.loads(analysis['parameters']),
                    'results': json.loads(analysis['results']),
                    'created_at': analysis['created_at'],
                    'execution_time': analysis['execution_time'],
                    'accuracy': analysis['accuracy'],
                    'total_return': analysis['total_return'],
                    'is_favorite': analysis['is_favorite'],
                    'notes': analysis['notes']
                },
                'trades': [dict(trade) for trade in trades]
            }
            
        except Exception as e:
            return {'success': False, 'message': f'获取详情失败: {str(e)}'}
        finally:
            conn.close()
    
    def toggle_favorite(self, analysis_id: int, user_id: int) -> bool:
        """切换收藏状态"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE analysis_history 
                SET is_favorite = NOT is_favorite 
                WHERE id = ? AND user_id = ?
            ''', (analysis_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"切换收藏失败: {e}")
            return False
        finally:
            conn.close()
    
    def add_analysis_note(self, analysis_id: int, user_id: int, note: str) -> bool:
        """添加分析备注"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE analysis_history 
                SET notes = ? 
                WHERE id = ? AND user_id = ?
            ''', (note, analysis_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"添加备注失败: {e}")
            return False
        finally:
            conn.close()

class FavoriteManager:
    """收藏夹管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def add_favorite(self, user_id: int, ticker: str, notes: str = "") -> bool:
        """添加收藏"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO user_favorites (user_id, ticker, notes)
                VALUES (?, ?, ?)
            ''', (user_id, ticker, notes))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"添加收藏失败: {e}")
            return False
        finally:
            conn.close()
    
    def remove_favorite(self, user_id: int, ticker: str) -> bool:
        """移除收藏"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM user_favorites WHERE user_id = ? AND ticker = ?
            ''', (user_id, ticker))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"移除收藏失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_favorites(self, user_id: int) -> List[Dict]:
        """获取用户收藏"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT ticker, added_at, notes FROM user_favorites 
                WHERE user_id = ? 
                ORDER BY added_at DESC
            ''', (user_id,))
            
            favorites = []
            for row in cursor.fetchall():
                favorites.append({
                    'ticker': row['ticker'],
                    'added_at': row['added_at'],
                    'notes': row['notes']
                })
            
            return favorites
            
        except Exception as e:
            print(f"获取收藏失败: {e}")
            return []
        finally:
            conn.close()

class FeedbackManager:
    """用户反馈管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def submit_feedback(self, user_id: int, feedback_type: str, 
                       content: str, rating: int = None) -> bool:
        """提交反馈"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO user_feedback (user_id, feedback_type, content, rating)
                VALUES (?, ?, ?, ?)
            ''', (user_id, feedback_type, content, rating))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"提交反馈失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_feedback(self, user_id: int) -> List[Dict]:
        """获取用户反馈历史"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT feedback_type, content, rating, created_at, status, admin_reply
                FROM user_feedback 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            ''', (user_id,))
            
            feedback_list = []
            for row in cursor.fetchall():
                feedback_list.append({
                    'feedback_type': row['feedback_type'],
                    'content': row['content'],
                    'rating': row['rating'],
                    'created_at': row['created_at'],
                    'status': row['status'],
                    'admin_reply': row['admin_reply']
                })
            
            return feedback_list
            
        except Exception as e:
            print(f"获取反馈失败: {e}")
            return []
        finally:
            conn.close()

    def get_all_feedback(self, status: str = None, limit: int = 200) -> List[Dict]:
        """管理员查看反馈列表（联表返回用户名）"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            limit = max(1, min(int(limit or 200), 1000))
            if status and status != 'all':
                cursor.execute('''
                    SELECT
                        f.id,
                        f.user_id,
                        COALESCE(u.username, '未知用户') AS username,
                        f.feedback_type,
                        f.content,
                        f.rating,
                        f.status,
                        f.admin_reply,
                        f.created_at
                    FROM user_feedback f
                    LEFT JOIN users u ON f.user_id = u.id
                    WHERE f.status = ?
                    ORDER BY f.created_at DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                cursor.execute('''
                    SELECT
                        f.id,
                        f.user_id,
                        COALESCE(u.username, '未知用户') AS username,
                        f.feedback_type,
                        f.content,
                        f.rating,
                        f.status,
                        f.admin_reply,
                        f.created_at
                    FROM user_feedback f
                    LEFT JOIN users u ON f.user_id = u.id
                    ORDER BY f.created_at DESC
                    LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"获取管理员反馈列表失败: {e}")
            return []
        finally:
            conn.close()

    def reply_feedback(self, feedback_id: int, admin_reply: str, new_status: str = 'resolved') -> bool:
        """管理员回复反馈并更新状态"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE user_feedback
                SET admin_reply = ?, status = ?
                WHERE id = ?
            ''', (admin_reply, new_status, feedback_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"回复反馈失败: {e}")
            return False
        finally:
            conn.close()

class PortfolioManager:
    """投资组合管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def get_user_portfolio(self, user_id: int) -> List[Dict]:
        """获取用户投资组合"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT stock_symbol, shares, avg_price, current_price, 
                       total_value, unrealized_pnl, last_updated
                FROM portfolios 
                WHERE user_id = ? AND shares > 0
                ORDER BY total_value DESC
            ''', (user_id,))
            
            portfolio = []
            for row in cursor.fetchall():
                portfolio.append({
                    'stock_symbol': row['stock_symbol'],
                    'shares': row['shares'],
                    'avg_price': row['avg_price'],
                    'current_price': row['current_price'],
                    'total_value': row['total_value'],
                    'unrealized_pnl': row['unrealized_pnl'],
                    'last_updated': row['last_updated']
                })
            
            return portfolio
            
        except Exception as e:
            print(f"获取投资组合失败: {e}")
            return []
        finally:
            conn.close()
    
    def update_portfolio(self, user_id: int, stock_symbol: str, 
                        action: str, price: float, quantity: int) -> bool:
        """更新投资组合"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # 获取当前持仓
            cursor.execute('''
                SELECT shares, avg_price FROM portfolios 
                WHERE user_id = ? AND stock_symbol = ?
            ''', (user_id, stock_symbol))
            
            current = cursor.fetchone()
            
            if action == 'buy':
                if current:
                    # 更新现有持仓
                    new_shares = current['shares'] + quantity
                    new_avg_price = ((current['shares'] * current['avg_price']) + 
                                   (quantity * price)) / new_shares
                    
                    cursor.execute('''
                        UPDATE portfolios 
                        SET shares = ?, avg_price = ?, current_price = ?,
                            total_value = ? * ?, last_updated = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND stock_symbol = ?
                    ''', (new_shares, new_avg_price, price, new_shares, price, 
                          user_id, stock_symbol))
                else:
                    # 新建持仓
                    cursor.execute('''
                        INSERT INTO portfolios 
                        (user_id, stock_symbol, shares, avg_price, current_price, 
                         total_value, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (user_id, stock_symbol, quantity, price, price, 
                          quantity * price))
            
            elif action == 'sell':
                if current and current['shares'] >= quantity:
                    new_shares = current['shares'] - quantity
                    if new_shares > 0:
                        cursor.execute('''
                            UPDATE portfolios 
                            SET shares = ?, current_price = ?,
                                total_value = ? * ?, last_updated = CURRENT_TIMESTAMP
                            WHERE user_id = ? AND stock_symbol = ?
                        ''', (new_shares, price, new_shares, price, 
                              user_id, stock_symbol))
                    else:
                        cursor.execute('''
                            DELETE FROM portfolios 
                            WHERE user_id = ? AND stock_symbol = ?
                        ''', (user_id, stock_symbol))
            
            # 记录交易
            total_amount = price * quantity
            cursor.execute('''
                INSERT INTO user_transactions 
                (user_id, stock_symbol, action, price, quantity, total_amount)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, stock_symbol, action, price, quantity, total_amount))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"更新投资组合失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_transactions(self, user_id: int, limit: int = 100) -> List[Dict]:
        """获取用户交易记录"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT stock_symbol, action, price, quantity, total_amount,
                       timestamp, strategy_used, notes
                FROM user_transactions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'stock_symbol': row['stock_symbol'],
                    'action': row['action'],
                    'price': row['price'],
                    'quantity': row['quantity'],
                    'total_amount': row['total_amount'],
                    'timestamp': row['timestamp'],
                    'strategy_used': row['strategy_used'],
                    'notes': row['notes']
                })
            
            return transactions
            
        except Exception as e:
            print(f"获取交易记录失败: {e}")
            return []
        finally:
            conn.close()

class UserSettingsManager:
    """用户设置管理类"""
    
    def __init__(self, db: EnhancedDatabase):
        self.db = db
    
    def get_user_setting(self, user_id: int, key: str) -> str:
        """获取用户设置"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT setting_value FROM user_settings 
                WHERE user_id = ? AND setting_key = ?
            ''', (user_id, key))
            
            result = cursor.fetchone()
            return result['setting_value'] if result else None
            
        except Exception as e:
            print(f"获取用户设置失败: {e}")
            return None
        finally:
            conn.close()
    
    def set_user_setting(self, user_id: int, key: str, value: str) -> bool:
        """设置用户设置"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO user_settings 
                (user_id, setting_key, setting_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, value))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            print(f"设置用户设置失败: {e}")
            return False
        finally:
            conn.close()

# 初始化增强数据库系统
def init_enhanced_database_system():
    """初始化增强数据库系统"""
    db = EnhancedDatabase()
    
    return {
        'db': db,
        'user_manager': UserManager(db),
        'analysis_manager': AnalysisManager(db),
        'favorite_manager': FavoriteManager(db),
        'feedback_manager': FeedbackManager(db),
        'portfolio_manager': PortfolioManager(db),
        'settings_manager': UserSettingsManager(db),
        'role_permission_manager': RolePermissionManager(db)
    }

if __name__ == "__main__":
    # 测试数据库系统
    system = init_enhanced_database_system()
    print("增强数据库系统初始化完成！")

