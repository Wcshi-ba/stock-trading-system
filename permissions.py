#!/usr/bin/env python3
"""
权限控制系统 - 基于角色的访问控制（RBAC）
"""

from functools import wraps
from flask import request, jsonify, session
from typing import Callable, Dict


class PermissionManager:
    """权限管理器 - 优先从数据库读取，无则回退到内置矩阵"""
    
    _db = None  # 由应用启动时注入
    
    PERMISSION_MATRIX = {
        'user': {
            'market_data.view': True,
            'market_data.export_basic': True,
            'features.view_preset': True,
            'features.visualization': True,
            'model.view_predictions': True,
            'strategy.simulate_trade': True,
            'strategy.view_backtest': True,
            'strategy.manual_stop_loss': True,
            'risk.view_personal': True,
            'risk.set_personal_threshold': True,
            'account.modify_own': True,
            'account.view_own_records': True,
        },
        'admin': {
            'market_data.manage_source': True,
            'market_data.audit_upload': True,
            'market_data.view_logs': True,
            'risk.config_global': True,
            'risk.monitor_platform': True,
            'risk.handle_alert': True,
            'risk.audit_rules': True,
            'account.create_user': True,
            'account.disable_user': True,
            'account.delete_user': True,
            'account.assign_role': True,
            'account.reset_password': True,
            'account.view_all_logs': True,
            'account.config_quota': True,
        },
        'guest': {
            'market_data.view': True,
            'features.view_preset': True,
            'model.view_predictions': True,
            'strategy.view_backtest': True,
        }
    }
    
    @classmethod
    def has_permission(cls, role: str, permission: str) -> bool:
        """检查角色是否拥有特定权限（优先读数据库）"""
        if cls._db:
            try:
                rpm = cls._db.get('role_permission_manager')
                if rpm:
                    perms = rpm.get_all_roles_permissions()
                    if role in perms and permission in perms[role]:
                        return bool(perms[role][permission])
            except Exception:
                pass
        if role not in cls.PERMISSION_MATRIX:
            return False
        return cls.PERMISSION_MATRIX[role].get(permission, False)
    
    @classmethod
    def get_user_permissions(cls, role: str) -> Dict[str, bool]:
        """获取角色的所有权限（优先读数据库）"""
        if cls._db:
            try:
                rpm = cls._db.get('role_permission_manager')
                if rpm:
                    all_perms = rpm.get_all_roles_permissions()
                    if role in all_perms:
                        return dict(all_perms[role])
            except Exception:
                pass
        perms = {}
        if role in cls.PERMISSION_MATRIX:
            perms.update(cls.PERMISSION_MATRIX[role])
        return perms


def require_permission(permission: str) -> Callable:
    """动态检查当前用户权限的装饰器"""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            role = session.get('role', 'guest')
            user_id = session.get('user_id')
            
            if role == 'guest' or not user_id:
                guest_allowed = ['market_data.view', 'features.view_preset', 
                                'model.view_predictions', 'strategy.view_backtest']
                if permission not in guest_allowed:
                    return jsonify(success=False, message='请先登录'), 401
            
            if not PermissionManager.has_permission(role, permission):
                return jsonify(
                    success=False,
                    message=f'权限不足：需要 {permission} 权限',
                    required_permission=permission,
                    current_role=role
                ), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def require_admin(f: Callable) -> Callable:
    """管理员专用装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        role = session.get('role')
        if role != 'admin':
            return jsonify(success=False, message='需要管理员权限'), 403
        return f(*args, **kwargs)
    return wrapper
