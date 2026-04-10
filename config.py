#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用配置 - 邮件、密钥等
使用前请将 MAIL_USERNAME 和 MAIL_PASSWORD 改为你自己的真实值。
  QQ邮箱：设置 → 账户 → 开启 SMTP → 获取授权码
  163邮箱：设置 → POP3/SMTP → 开启 → 获取授权码
"""

import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string-enhanced-trading-2024'

    # ========== 邮件 SMTP 配置（只需改这一处）==========
    MAIL_SERVER   = os.environ.get('MAIL_SERVER')   or 'smtp.qq.com'
    MAIL_PORT     = int(os.environ.get('MAIL_PORT')  or 587)
    MAIL_USE_TLS  = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME') or 'your-qq@qq.com'           # ← 改成你的发件邮箱
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD') or 'your-authorization-code'  # ← 改成邮箱授权码
    MAIL_DEFAULT_SENDER = MAIL_USERNAME

    # 是否真正发邮件（False 时仅打印到控制台，方便本地调试）
    MAIL_ENABLED = MAIL_USERNAME != 'your-qq@qq.com'
