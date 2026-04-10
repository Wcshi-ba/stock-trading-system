#!/usr/bin/env python3
"""
通用邮件服务 —— 供密码重置、止盈预警等场景使用
SMTP 参数统一从 config.py 读取。
"""

import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict
import sqlite3

from config import Config


class MailService:
    """通用邮件发送服务"""

    def __init__(self, db_path: str = "enhanced_trading_system.db"):
        self.db_path = db_path
        self.host     = Config.MAIL_SERVER
        self.port     = Config.MAIL_PORT
        self.use_tls  = Config.MAIL_USE_TLS
        self.username = Config.MAIL_USERNAME
        self.password = Config.MAIL_PASSWORD
        self.sender   = Config.MAIL_DEFAULT_SENDER
        self.enabled  = Config.MAIL_ENABLED

    # ------------------------------------------------------------------
    # 底层：发送任意 HTML 邮件
    # ------------------------------------------------------------------
    def send_html_email(self, to_email: str, subject: str, html_body: str) -> bool:
        """发送 HTML 邮件，返回是否成功"""
        if not self.enabled:
            print(f"[邮件-演示模式] → {to_email}")
            print(f"  主题: {subject}")
            print(f"  （SMTP 未配置，邮件内容仅打印到控制台）")
            return True

        try:
            msg = MIMEMultipart()
            msg['From']    = self.sender
            msg['To']      = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(html_body, 'html', 'utf-8'))

            server = smtplib.SMTP(self.host, self.port, timeout=10)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            print(f"[邮件] 发送成功 → {to_email}  主题: {subject}")
            return True
        except Exception as e:
            print(f"[邮件] 发送失败 → {to_email}  错误: {e}")
            return False

    # ------------------------------------------------------------------
    # 场景 1：密码重置码邮件
    # ------------------------------------------------------------------
    def send_reset_code_email(self, to_email: str, username: str, reset_code: str) -> bool:
        subject = '智能股票交易系统 - 密码重置码'
        html = f"""
        <html><body style="font-family:Arial,sans-serif;color:#333;">
        <div style="max-width:600px;margin:0 auto;padding:20px;">
            <h2 style="color:#2c3e50;text-align:center;">智能股票交易系统</h2>
            <div style="background:#f8f9fa;padding:20px;border-radius:8px;margin:20px 0;">
                <h3 style="color:#495057;margin-top:0;">密码重置</h3>
                <p>用户名 <b>{username}</b>，您好！</p>
                <p>您申请了密码重置，请使用以下重置码完成操作：</p>
                <div style="text-align:center;margin:30px 0;">
                    <span style="font-size:28px;font-weight:bold;color:#007bff;
                                background:#e3f2fd;padding:15px 30px;border-radius:8px;
                                letter-spacing:3px;">{reset_code}</span>
                </div>
                <p><b>有效期：10 分钟</b>，过期后请重新申请。</p>
                <p>如果这不是您的操作，请忽略此邮件并确保账户安全。</p>
            </div>
            <div style="text-align:center;color:#6c757d;font-size:12px;margin-top:30px;">
                <p>此邮件由系统自动发送，请勿回复</p>
            </div>
        </div>
        </body></html>"""
        return self.send_html_email(to_email, subject, html)

    # ------------------------------------------------------------------
    # 场景 2：止盈 / 止损预警邮件
    # ------------------------------------------------------------------
    def send_stock_alert_email(self, to_email: str, username: str,
                               ticker: str, current_price: float,
                               target_price: float, alert_type: str = '止盈') -> bool:
        direction = '高于' if alert_type == '止盈' else '低于'
        color = '#27ae60' if alert_type == '止盈' else '#e74c3c'
        subject = f'智能股票交易系统 - {ticker} 已触达{alert_type}线'
        html = f"""
        <html><body style="font-family:Arial,sans-serif;color:#333;">
        <div style="max-width:600px;margin:0 auto;padding:20px;">
            <h2 style="color:#2c3e50;text-align:center;">智能股票交易系统</h2>
            <div style="background:#f8f9fa;padding:20px;border-radius:8px;margin:20px 0;">
                <h3 style="color:{color};margin-top:0;">{alert_type}预警</h3>
                <p>{username}，您好！</p>
                <p>您关注的 <b>{ticker}</b> 当前价格已{direction}预设{alert_type}线：</p>
                <table style="width:100%;border-collapse:collapse;margin:20px 0;">
                    <tr style="background:#e3f2fd;">
                        <td style="padding:12px;border:1px solid #dee2e6;">股票代码</td>
                        <td style="padding:12px;border:1px solid #dee2e6;font-weight:bold;">{ticker}</td>
                    </tr>
                    <tr>
                        <td style="padding:12px;border:1px solid #dee2e6;">当前价格</td>
                        <td style="padding:12px;border:1px solid #dee2e6;font-weight:bold;color:{color};">${current_price:.2f}</td>
                    </tr>
                    <tr style="background:#e3f2fd;">
                        <td style="padding:12px;border:1px solid #dee2e6;">{alert_type}目标价</td>
                        <td style="padding:12px;border:1px solid #dee2e6;">${target_price:.2f}</td>
                    </tr>
                </table>
                <p>请及时查看并做出交易决策。</p>
            </div>
            <div style="text-align:center;color:#6c757d;font-size:12px;margin-top:30px;">
                <p>此邮件由系统自动发送，请勿回复</p>
            </div>
        </div>
        </body></html>"""
        return self.send_html_email(to_email, subject, html)


# 全局单例
mail_service = MailService()


# ---- 兼容旧代码的便捷函数 ----
def send_verification_code(email: str) -> Dict[str, any]:
    code = ''.join(random.choices(string.digits, k=6))
    ok = mail_service.send_html_email(
        email,
        '智能股票交易系统 - 验证码',
        f'<p>您的验证码是 <b>{code}</b>，10 分钟内有效。</p>'
    )
    return {'success': ok, 'message': '验证码已发送' if ok else '发送失败', 'code': code}


def verify_email_code(email: str, code: str) -> Dict[str, any]:
    return {'success': False, 'message': '验证码功能已简化，请使用密码重置流程'}


if __name__ == "__main__":
    print("邮件服务自检...")
    print(f"  SMTP:    {Config.MAIL_SERVER}:{Config.MAIL_PORT}")
    print(f"  发件人:  {Config.MAIL_USERNAME}")
    print(f"  已启用:  {Config.MAIL_ENABLED}")
    if not Config.MAIL_ENABLED:
        print("  ⚠ 当前为演示模式，邮件仅打印到控制台。")
        print("  请在 config.py 中填入真实 SMTP 信息后重试。")
    else:
        test = input("输入测试收件邮箱（回车跳过）：").strip()
        if test:
            ok = mail_service.send_reset_code_email(test, 'TestUser', 'RESET-TEST-12345678')
            print(f"发送结果: {'成功' if ok else '失败'}")
