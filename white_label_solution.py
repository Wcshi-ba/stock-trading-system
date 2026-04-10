#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白标解决方案
White-Label Solution
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class WhiteLabelManager:
    """白标管理器"""
    
    def __init__(self):
        self.tenants = {}
        self.branding_configs = {}
        self.custom_features = {}
        self.api_customizations = {}
        self.theme_configs = {}
        
    def create_tenant(self, tenant_name: str, domain: str, 
                     admin_email: str, **kwargs) -> str:
        """创建租户"""
        tenant_id = str(uuid.uuid4())
        
        tenant_data = {
            'id': tenant_id,
            'name': tenant_name,
            'domain': domain,
            'admin_email': admin_email,
            'created_at': datetime.now(),
            'is_active': True,
            'subscription_plan': kwargs.get('subscription_plan', 'basic'),
            'custom_domain': kwargs.get('custom_domain', None),
            'metadata': kwargs
        }
        
        self.tenants[tenant_id] = tenant_data
        
        # 初始化默认配置
        self._initialize_tenant_config(tenant_id)
        
        return tenant_id
    
    def _initialize_tenant_config(self, tenant_id: str):
        """初始化租户配置"""
        # 默认品牌配置
        self.branding_configs[tenant_id] = {
            'logo_url': None,
            'favicon_url': None,
            'primary_color': '#667eea',
            'secondary_color': '#764ba2',
            'accent_color': '#f093fb',
            'background_color': '#ffffff',
            'text_color': '#333333',
            'font_family': 'Arial, sans-serif',
            'company_name': self.tenants[tenant_id]['name'],
            'tagline': '智能投资，专业分析',
            'contact_email': self.tenants[tenant_id]['admin_email'],
            'contact_phone': None,
            'address': None,
            'social_links': {}
        }
        
        # 默认主题配置
        self.theme_configs[tenant_id] = {
            'theme_name': 'default',
            'layout': 'sidebar',
            'header_style': 'fixed',
            'sidebar_collapsed': False,
            'dark_mode': False,
            'animations': True,
            'custom_css': None,
            'custom_js': None
        }
        
        # 默认功能配置
        self.custom_features[tenant_id] = {
            'features': {
                'portfolio_management': True,
                'risk_analysis': True,
                'trading_signals': True,
                'market_data': True,
                'reports': True,
                'alerts': True,
                'api_access': False,
                'white_label': True,
                'custom_branding': True,
                'multi_user': True
            },
            'limits': {
                'max_users': 10,
                'max_portfolios': 5,
                'api_calls_per_hour': 1000,
                'storage_gb': 1,
                'data_retention_days': 365
            },
            'permissions': {
                'can_customize_branding': True,
                'can_manage_users': True,
                'can_access_api': False,
                'can_export_data': True,
                'can_import_data': True
            }
        }
        
        # 默认API定制
        self.api_customizations[tenant_id] = {
            'base_url': f"https://api.{self.tenants[tenant_id]['domain']}",
            'version': 'v1',
            'rate_limits': {
                'requests_per_minute': 60,
                'requests_per_hour': 1000,
                'requests_per_day': 10000
            },
            'authentication': {
                'type': 'api_key',
                'required': True,
                'key_header': 'X-API-Key'
            },
            'endpoints': {
                'enabled': [
                    '/market/stocks',
                    '/analysis/predict',
                    '/portfolio/overview',
                    '/user/profile'
                ],
                'disabled': [
                    '/admin/users',
                    '/admin/system',
                    '/admin/analytics'
                ]
            },
            'response_format': {
                'include_metadata': True,
                'include_timestamps': True,
                'date_format': 'ISO',
                'timezone': 'UTC'
            }
        }
    
    def update_branding(self, tenant_id: str, branding_config: Dict) -> bool:
        """更新品牌配置"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        # 合并配置
        current_config = self.branding_configs.get(tenant_id, {})
        current_config.update(branding_config)
        self.branding_configs[tenant_id] = current_config
        
        return True
    
    def update_theme(self, tenant_id: str, theme_config: Dict) -> bool:
        """更新主题配置"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        # 合并配置
        current_config = self.theme_configs.get(tenant_id, {})
        current_config.update(theme_config)
        self.theme_configs[tenant_id] = current_config
        
        return True
    
    def update_features(self, tenant_id: str, features_config: Dict) -> bool:
        """更新功能配置"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        # 合并配置
        current_config = self.custom_features.get(tenant_id, {})
        
        if 'features' in features_config:
            current_config['features'].update(features_config['features'])
        if 'limits' in features_config:
            current_config['limits'].update(features_config['limits'])
        if 'permissions' in features_config:
            current_config['permissions'].update(features_config['permissions'])
        
        self.custom_features[tenant_id] = current_config
        
        return True
    
    def update_api_customization(self, tenant_id: str, api_config: Dict) -> bool:
        """更新API定制"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        # 合并配置
        current_config = self.api_customizations.get(tenant_id, {})
        current_config.update(api_config)
        self.api_customizations[tenant_id] = current_config
        
        return True
    
    def generate_custom_css(self, tenant_id: str) -> str:
        """生成自定义CSS"""
        if tenant_id not in self.branding_configs:
            return ""
        
        branding = self.branding_configs[tenant_id]
        theme = self.theme_configs.get(tenant_id, {})
        
        css = f"""
        /* 自定义样式 - {branding.get('company_name', '')} */
        :root {{
            --primary-color: {branding.get('primary_color', '#667eea')};
            --secondary-color: {branding.get('secondary_color', '#764ba2')};
            --accent-color: {branding.get('accent_color', '#f093fb')};
            --background-color: {branding.get('background_color', '#ffffff')};
            --text-color: {branding.get('text_color', '#333333')};
            --font-family: {branding.get('font_family', 'Arial, sans-serif')};
        }}
        
        body {{
            font-family: var(--font-family);
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
        }}
        
        .btn-primary {{
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }}
        
        .btn-primary:hover {{
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
        }}
        
        .card {{
            border-left: 4px solid var(--accent-color);
        }}
        
        .sidebar {{
            background-color: var(--background-color);
            border-right: 1px solid #e0e0e0;
        }}
        
        .logo {{
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .tagline {{
            color: var(--text-color);
            opacity: 0.8;
        }}
        """
        
        # 添加自定义CSS
        if branding.get('custom_css'):
            css += f"\n/* 自定义CSS */\n{branding['custom_css']}"
        
        return css
    
    def generate_custom_js(self, tenant_id: str) -> str:
        """生成自定义JavaScript"""
        if tenant_id not in self.tenants:
            return ""
        
        tenant = self.tenants[tenant_id]
        branding = self.branding_configs.get(tenant_id, {})
        
        js = f"""
        // 自定义JavaScript - {branding.get('company_name', '')}
        window.tenantConfig = {{
            tenantId: '{tenant_id}',
            companyName: '{branding.get('company_name', '')}',
            tagline: '{branding.get('tagline', '')}',
            contactEmail: '{branding.get('contact_email', '')}',
            contactPhone: '{branding.get('contact_phone', '')}',
            primaryColor: '{branding.get('primary_color', '#667eea')}',
            secondaryColor: '{branding.get('secondary_color', '#764ba2')}'
        }};
        
        // 初始化自定义功能
        document.addEventListener('DOMContentLoaded', function() {{
            // 更新页面标题
            document.title = '{branding.get('company_name', '')} - 智能投资平台';
            
            // 更新logo
            const logo = document.querySelector('.logo');
            if (logo) {{
                logo.textContent = '{branding.get('company_name', '')}';
            }}
            
            // 更新标语
            const tagline = document.querySelector('.tagline');
            if (tagline) {{
                tagline.textContent = '{branding.get('tagline', '')}';
            }}
            
            // 应用主题
            applyTheme();
        }});
        
        function applyTheme() {{
            const root = document.documentElement;
            root.style.setProperty('--primary-color', window.tenantConfig.primaryColor);
            root.style.setProperty('--secondary-color', window.tenantConfig.secondaryColor);
        }}
        """
        
        # 添加自定义JavaScript
        if branding.get('custom_js'):
            js += f"\n/* 自定义JavaScript */\n{branding['custom_js']}"
        
        return js
    
    def generate_config_file(self, tenant_id: str, format: str = 'json') -> str:
        """生成配置文件"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        config = {
            'tenant': self.tenants[tenant_id],
            'branding': self.branding_configs.get(tenant_id, {}),
            'theme': self.theme_configs.get(tenant_id, {}),
            'features': self.custom_features.get(tenant_id, {}),
            'api': self.api_customizations.get(tenant_id, {}),
            'generated_at': datetime.now().isoformat()
        }
        
        if format == 'json':
            # 处理datetime对象
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            
            return json.dumps(config, indent=2, ensure_ascii=False, default=json_serializer)
        elif format == 'yaml':
            import yaml
            return yaml.dump(config, default_flow_style=False, allow_unicode=True)
        else:
            return str(config)
    
    def get_tenant_dashboard_data(self, tenant_id: str) -> Dict:
        """获取租户仪表板数据"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        tenant = self.tenants[tenant_id]
        features = self.custom_features.get(tenant_id, {})
        
        # 模拟仪表板数据
        dashboard_data = {
            'tenant_info': {
                'name': tenant['name'],
                'domain': tenant['domain'],
                'subscription_plan': tenant['subscription_plan'],
                'created_at': tenant['created_at']
            },
            'usage_stats': {
                'active_users': np.random.randint(1, features['limits']['max_users']),
                'portfolios_created': np.random.randint(0, features['limits']['max_portfolios']),
                'api_calls_today': np.random.randint(0, features['limits']['api_calls_per_hour']),
                'storage_used_gb': np.random.uniform(0, features['limits']['storage_gb'])
            },
            'features_enabled': features['features'],
            'limits': features['limits'],
            'branding': self.branding_configs.get(tenant_id, {}),
            'theme': self.theme_configs.get(tenant_id, {})
        }
        
        return dashboard_data
    
    def validate_tenant_config(self, tenant_id: str) -> Dict:
        """验证租户配置"""
        if tenant_id not in self.tenants:
            raise ValueError(f"租户不存在: {tenant_id}")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 验证品牌配置
        branding = self.branding_configs.get(tenant_id, {})
        if not branding.get('company_name'):
            validation_results['errors'].append('公司名称不能为空')
            validation_results['valid'] = False
        
        if not branding.get('contact_email'):
            validation_results['warnings'].append('联系邮箱未设置')
        
        # 验证主题配置
        theme = self.theme_configs.get(tenant_id, {})
        if not theme.get('theme_name'):
            validation_results['warnings'].append('主题名称未设置')
        
        # 验证功能配置
        features = self.custom_features.get(tenant_id, {})
        if not features.get('features'):
            validation_results['errors'].append('功能配置缺失')
            validation_results['valid'] = False
        
        # 验证API配置
        api_config = self.api_customizations.get(tenant_id, {})
        if not api_config.get('base_url'):
            validation_results['warnings'].append('API基础URL未设置')
        
        return validation_results
    
    def get_all_tenants(self) -> List[Dict]:
        """获取所有租户"""
        tenants_list = []
        for tenant_id, tenant in self.tenants.items():
            tenant_info = {
                'id': tenant_id,
                'name': tenant['name'],
                'domain': tenant['domain'],
                'subscription_plan': tenant['subscription_plan'],
                'is_active': tenant['is_active'],
                'created_at': tenant['created_at'],
                'features_count': len(self.custom_features.get(tenant_id, {}).get('features', {})),
                'has_custom_branding': bool(self.branding_configs.get(tenant_id, {}).get('logo_url'))
            }
            tenants_list.append(tenant_info)
        
        return tenants_list

def test_white_label_solution():
    """测试白标解决方案"""
    print("测试白标解决方案...")
    
    # 初始化白标管理器
    wl_manager = WhiteLabelManager()
    
    # 创建租户
    print("1. 创建租户...")
    tenant1_id = wl_manager.create_tenant(
        '投资公司A', 'invest-a.com', 'admin@invest-a.com',
        subscription_plan='premium',
        custom_domain='trading.invest-a.com'
    )
    print(f"   创建租户: {tenant1_id}")
    
    tenant2_id = wl_manager.create_tenant(
        '金融科技B', 'fintech-b.com', 'admin@fintech-b.com',
        subscription_plan='basic'
    )
    print(f"   创建租户: {tenant2_id}")
    
    # 更新品牌配置
    print("2. 更新品牌配置...")
    branding_config = {
        'logo_url': 'https://example.com/logo.png',
        'primary_color': '#ff6b6b',
        'secondary_color': '#4ecdc4',
        'company_name': '投资公司A',
        'tagline': '专业投资，智能分析',
        'contact_email': 'contact@invest-a.com',
        'contact_phone': '+86-400-123-4567'
    }
    wl_manager.update_branding(tenant1_id, branding_config)
    print("   品牌配置更新完成")
    
    # 更新主题配置
    print("3. 更新主题配置...")
    theme_config = {
        'theme_name': 'corporate',
        'layout': 'topbar',
        'dark_mode': False,
        'animations': True
    }
    wl_manager.update_theme(tenant1_id, theme_config)
    print("   主题配置更新完成")
    
    # 更新功能配置
    print("4. 更新功能配置...")
    features_config = {
        'features': {
            'api_access': True,
            'custom_branding': True,
            'multi_user': True
        },
        'limits': {
            'max_users': 50,
            'api_calls_per_hour': 5000
        }
    }
    wl_manager.update_features(tenant1_id, features_config)
    print("   功能配置更新完成")
    
    # 更新API定制
    print("5. 更新API定制...")
    api_config = {
        'base_url': 'https://api.invest-a.com',
        'rate_limits': {
            'requests_per_minute': 100,
            'requests_per_hour': 5000
        },
        'endpoints': {
            'enabled': [
                '/market/stocks',
                '/analysis/predict',
                '/portfolio/overview',
                '/user/profile',
                '/admin/dashboard'
            ]
        }
    }
    wl_manager.update_api_customization(tenant1_id, api_config)
    print("   API定制更新完成")
    
    # 生成自定义CSS
    print("6. 生成自定义CSS...")
    custom_css = wl_manager.generate_custom_css(tenant1_id)
    print(f"   CSS长度: {len(custom_css)} 字符")
    
    # 生成自定义JavaScript
    print("7. 生成自定义JavaScript...")
    custom_js = wl_manager.generate_custom_js(tenant1_id)
    print(f"   JavaScript长度: {len(custom_js)} 字符")
    
    # 生成配置文件
    print("8. 生成配置文件...")
    config_json = wl_manager.generate_config_file(tenant1_id, 'json')
    print(f"   配置文件长度: {len(config_json)} 字符")
    
    # 获取仪表板数据
    print("9. 获取仪表板数据...")
    dashboard_data = wl_manager.get_tenant_dashboard_data(tenant1_id)
    print(f"   活跃用户: {dashboard_data['usage_stats']['active_users']}")
    print(f"   创建的投资组合: {dashboard_data['usage_stats']['portfolios_created']}")
    print(f"   今日API调用: {dashboard_data['usage_stats']['api_calls_today']}")
    
    # 验证配置
    print("10. 验证配置...")
    validation = wl_manager.validate_tenant_config(tenant1_id)
    print(f"   配置有效: {'是' if validation['valid'] else '否'}")
    print(f"   错误数量: {len(validation['errors'])}")
    print(f"   警告数量: {len(validation['warnings'])}")
    
    # 获取所有租户
    print("11. 获取所有租户...")
    all_tenants = wl_manager.get_all_tenants()
    print(f"   总租户数量: {len(all_tenants)}")
    for tenant in all_tenants:
        print(f"   - {tenant['name']} ({tenant['domain']}) - {tenant['subscription_plan']}")
    
    print("白标解决方案测试完成！")
    return True

if __name__ == "__main__":
    test_white_label_solution()
