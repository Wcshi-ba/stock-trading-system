#!/usr/bin/env python3
"""
PDF报告生成器
支持一键导出包含4图+10项指标的专业分析报告
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PDFReportGenerator:
    """PDF报告生成器"""
    
    def __init__(self, output_dir: str = "results/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 报告配置
        self.report_config = {
            'title': '智能股票交易系统分析报告',
            'subtitle': '基于LSTM-进化策略模型的预测与交易分析',
            'company': '智能股票交易系统',
            'version': 'v1.0',
            'footer': '本报告仅供参考，投资有风险，入市需谨慎'
        }
    
    def generate_report(self, ticker: str, analysis_data: Dict, 
                       chart_paths: Dict[str, str], 
                       metrics: Dict[str, Any]) -> str:
        """
        生成完整的PDF报告
        
        Args:
            ticker: 股票代码
            analysis_data: 分析数据
            chart_paths: 图表路径字典
            metrics: 指标数据
        
        Returns:
            生成的PDF文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_analysis_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建PDF
        with PdfPages(filepath) as pdf:
            # 封面页
            self._create_cover_page(pdf, ticker, analysis_data)
            
            # 执行摘要
            self._create_executive_summary(pdf, ticker, metrics)
            
            # 技术分析图表
            self._create_technical_charts(pdf, chart_paths)
            
            # 交易策略分析
            self._create_strategy_analysis(pdf, ticker, analysis_data, metrics)
            
            # 风险评估
            self._create_risk_assessment(pdf, metrics)
            
            # 投资建议
            self._create_investment_recommendation(pdf, ticker, metrics)
            
            # 附录
            self._create_appendix(pdf, ticker, analysis_data)
        
        print(f"PDF报告已生成: {filepath}")
        return filepath
    
    def _create_cover_page(self, pdf: PdfPages, ticker: str, analysis_data: Dict):
        """创建封面页"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.8, self.report_config['title'], 
                ha='center', va='center', fontsize=24, fontweight='bold',
                color='#2c3e50')
        
        # 副标题
        ax.text(0.5, 0.75, self.report_config['subtitle'], 
                ha='center', va='center', fontsize=14, 
                color='#7f8c8d', style='italic')
        
        # 股票信息
        ax.text(0.5, 0.65, f"股票代码: {ticker}", 
                ha='center', va='center', fontsize=18, fontweight='bold',
                color='#3498db')
        
        # 分析时间
        analysis_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        ax.text(0.5, 0.6, f"分析时间: {analysis_time}", 
                ha='center', va='center', fontsize=12,
                color='#7f8c8d')
        
        # 关键指标摘要
        if 'metrics' in analysis_data:
            metrics = analysis_data['metrics']
            summary_text = f"""
关键指标摘要:
• 预测准确率: {metrics.get('accuracy', 0):.1%}
• 年化收益率: {metrics.get('annual_return', 0):.1%}
• 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
• 最大回撤: {metrics.get('max_drawdown', 0):.1%}
            """
            ax.text(0.5, 0.45, summary_text, 
                    ha='center', va='center', fontsize=12,
                    color='#2c3e50')
        
        # 公司信息
        ax.text(0.5, 0.1, f"{self.report_config['company']} {self.report_config['version']}", 
                ha='center', va='center', fontsize=10,
                color='#95a5a6')
        
        # 免责声明
        ax.text(0.5, 0.05, self.report_config['footer'], 
                ha='center', va='center', fontsize=8,
                color='#e74c3c', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_executive_summary(self, pdf: PdfPages, ticker: str, metrics: Dict):
        """创建执行摘要"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, "执行摘要", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color='#2c3e50')
        
        # 摘要内容
        summary_content = f"""
本报告基于LSTM神经网络和深度进化策略对{ticker}股票进行了全面的技术分析和交易策略评估。

主要发现:
1. 模型预测准确率达到{metrics.get('accuracy', 0):.1%}，显示出良好的预测能力
2. 交易策略年化收益率为{metrics.get('annual_return', 0):.1%}，超越市场基准
3. 夏普比率为{metrics.get('sharpe_ratio', 0):.2f}，风险调整后收益表现优秀
4. 最大回撤为{metrics.get('max_drawdown', 0):.1%}，风险控制良好
5. 交易胜率为{metrics.get('win_rate', 0):.1%}，策略稳定性较高

技术分析:
• 使用了20种技术指标进行特征工程
• LSTM模型捕捉了股价的长期依赖关系
• 深度进化策略优化了交易决策过程
• 四层风控体系确保了投资安全

投资建议:
基于分析结果，建议投资者{self._get_investment_advice(metrics)}。
        """
        
        ax.text(0.05, 0.85, summary_content, 
                ha='left', va='top', fontsize=11,
                color='#2c3e50', wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_technical_charts(self, pdf: PdfPages, chart_paths: Dict[str, str]):
        """创建技术分析图表页"""
        # 图表1: 价格预测
        if 'prediction' in chart_paths and os.path.exists(chart_paths['prediction']):
            self._add_chart_page(pdf, chart_paths['prediction'], "价格预测分析")
        
        # 图表2: 训练损失
        if 'loss' in chart_paths and os.path.exists(chart_paths['loss']):
            self._add_chart_page(pdf, chart_paths['loss'], "模型训练过程")
        
        # 图表3: 累积收益
        if 'earnings' in chart_paths and os.path.exists(chart_paths['earnings']):
            self._add_chart_page(pdf, chart_paths['earnings'], "累积收益分析")
        
        # 图表4: 交易信号
        if 'trades' in chart_paths and os.path.exists(chart_paths['trades']):
            self._add_chart_page(pdf, chart_paths['trades'], "交易信号分析")
    
    def _add_chart_page(self, pdf: PdfPages, chart_path: str, title: str):
        """添加图表页面"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, title, 
                ha='center', va='center', fontsize=18, fontweight='bold',
                color='#2c3e50')
        
        # 加载并显示图表
        try:
            from PIL import Image
            img = Image.open(chart_path)
            
            # 计算图片尺寸和位置
            img_width = 0.8
            img_height = 0.7
            img_x = (1 - img_width) / 2
            img_y = 0.2
            
            # 显示图片
            ax.imshow(img, extent=[img_x, img_x + img_width, img_y, img_y + img_height], 
                     aspect='auto', zorder=1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"图表加载失败: {str(e)}", 
                    ha='center', va='center', fontsize=12,
                    color='#e74c3c')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_strategy_analysis(self, pdf: PdfPages, ticker: str, 
                                 analysis_data: Dict, metrics: Dict):
        """创建交易策略分析页"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, "交易策略分析", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color='#2c3e50')
        
        # 策略描述
        strategy_content = f"""
策略概述:
本系统采用LSTM神经网络进行价格预测，结合深度进化策略进行交易决策。

LSTM预测模型:
• 输入特征: 20种技术指标（MA、RSI、MACD等）
• 网络结构: 多层LSTM + Dropout层
• 训练数据: {ticker}历史价格数据
• 预测目标: 次日收益率

深度进化策略:
• 种群大小: 128个个体
• 学习率: 0.02
• 扰动标准差: 0.1
• 迭代次数: 300代
• 动作空间: 买入/卖出/持有

策略表现:
• 总交易次数: {metrics.get('total_trades', 0)}次
• 胜率: {metrics.get('win_rate', 0):.1%}
• 平均收益: {metrics.get('avg_return', 0):.2%}
• 最大单次收益: {metrics.get('max_single_return', 0):.2%}
• 最大单次亏损: {metrics.get('max_single_loss', 0):.2%}

风险控制:
• VaR (95%): {metrics.get('var_95', 0):.2%}
• Kelly杠杆: {metrics.get('kelly_leverage', 0):.2f}
• 波动率仓位: {metrics.get('volatility_position', 0):.2f}
• 追踪止损: {metrics.get('stop_loss', 0):.2%}
        """
        
        ax.text(0.05, 0.85, strategy_content, 
                ha='left', va='top', fontsize=10,
                color='#2c3e50')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_risk_assessment(self, pdf: PdfPages, metrics: Dict):
        """创建风险评估页"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, "风险评估", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color='#2c3e50')
        
        # 风险指标表格
        risk_data = [
            ['风险指标', '数值', '评级', '说明'],
            ['最大回撤', f"{metrics.get('max_drawdown', 0):.2%}", 
             self._get_risk_rating(metrics.get('max_drawdown', 0), 'drawdown'),
             '投资期间的最大损失'],
            ['夏普比率', f"{metrics.get('sharpe_ratio', 0):.2f}", 
             self._get_risk_rating(metrics.get('sharpe_ratio', 0), 'sharpe'),
             '风险调整后收益'],
            ['波动率', f"{metrics.get('volatility', 0):.2%}", 
             self._get_risk_rating(metrics.get('volatility', 0), 'volatility'),
             '收益波动程度'],
            ['VaR (95%)', f"{metrics.get('var_95', 0):.2%}", 
             self._get_risk_rating(metrics.get('var_95', 0), 'var'),
             '95%置信度下的最大损失'],
            ['胜率', f"{metrics.get('win_rate', 0):.1%}", 
             self._get_risk_rating(metrics.get('win_rate', 0), 'win_rate'),
             '盈利交易占比']
        ]
        
        # 绘制表格
        y_start = 0.8
        y_step = 0.1
        
        for i, row in enumerate(risk_data):
            y_pos = y_start - i * y_step
            
            # 表头
            if i == 0:
                for j, cell in enumerate(row):
                    x_pos = 0.1 + j * 0.2
                    ax.text(x_pos, y_pos, cell, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='#2c3e50')
            else:
                # 数据行
                for j, cell in enumerate(row):
                    x_pos = 0.1 + j * 0.2
                    color = '#e74c3c' if j == 2 and '高' in str(cell) else '#2c3e50'
                    ax.text(x_pos, y_pos, cell, ha='center', va='center', 
                           fontsize=9, color=color)
        
        # 风险建议
        risk_advice = f"""
风险控制建议:
1. 建议仓位: {self._get_position_suggestion(metrics)}
2. 止损设置: {metrics.get('stop_loss', 0):.1%}
3. 风险等级: {self._get_overall_risk_level(metrics)}
4. 适合投资者: {self._get_suitable_investors(metrics)}

注意事项:
• 本分析基于历史数据，未来表现可能不同
• 建议结合基本面分析进行投资决策
• 定期调整仓位和止损设置
• 分散投资以降低风险
        """
        
        ax.text(0.05, 0.3, risk_advice, 
                ha='left', va='top', fontsize=10,
                color='#2c3e50')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_investment_recommendation(self, pdf: PdfPages, ticker: str, metrics: Dict):
        """创建投资建议页"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, "投资建议", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color='#2c3e50')
        
        # 投资建议
        recommendation = f"""
基于对{ticker}的全面分析，我们提供以下投资建议:

投资评级: {self._get_investment_rating(metrics)}

短期建议 (1-3个月):
{self._get_short_term_advice(metrics)}

中期建议 (3-12个月):
{self._get_medium_term_advice(metrics)}

长期建议 (1年以上):
{self._get_long_term_advice(metrics)}

关键风险点:
• 市场系统性风险
• 个股基本面变化
• 技术指标失效风险
• 流动性风险

操作建议:
• 建议仓位: {self._get_position_suggestion(metrics)}
• 入场时机: {self._get_entry_timing(metrics)}
• 止损位: {metrics.get('stop_loss', 0):.1%}
• 止盈位: {metrics.get('take_profit', 0):.1%}

免责声明:
本报告仅供参考，不构成投资建议。投资者应根据自身情况做出投资决策，
并承担相应风险。投资有风险，入市需谨慎。
        """
        
        ax.text(0.05, 0.85, recommendation, 
                ha='left', va='top', fontsize=10,
                color='#2c3e50')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_appendix(self, pdf: PdfPages, ticker: str, analysis_data: Dict):
        """创建附录"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.95, "附录", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color='#2c3e50')
        
        # 技术指标说明
        appendix_content = f"""
技术指标说明:

趋势指标:
• MA5/MA10/MA20: 5日、10日、20日移动平均线
• SMA: 简单移动平均线
• VWAP: 成交量加权平均价格

动量指标:
• RSI: 相对强弱指数，衡量超买超卖
• MACD: 指数平滑移动平均线
• ATR: 平均真实波幅

波动率指标:
• 布林带: 价格波动区间
• 标准差: 价格波动程度

成交量指标:
• Volume: 成交量
• 相对表现: 相对于基准的表现

模型参数:
• LSTM隐藏层: 64个神经元
• 训练轮数: 50-500轮
• 学习率: 0.001
• 批次大小: 32
• 窗口大小: 30天

数据来源:
• 价格数据: Yahoo Finance
• 技术指标: 系统自动计算
• 分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

联系方式:
• 系统版本: v1.0
• 技术支持: 智能股票交易系统
• 更新频率: 实时更新
        """
        
        ax.text(0.05, 0.85, appendix_content, 
                ha='left', va='top', fontsize=9,
                color='#2c3e50')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # 辅助方法
    def _get_investment_advice(self, metrics: Dict) -> str:
        """获取投资建议"""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        
        if sharpe > 1.5 and max_dd < 0.1:
            return "积极关注，适合中长期投资"
        elif sharpe > 1.0 and max_dd < 0.15:
            return "谨慎乐观，适合适度配置"
        elif sharpe > 0.5:
            return "保持观望，等待更好时机"
        else:
            return "风险较高，建议规避"
    
    def _get_risk_rating(self, value: float, metric_type: str) -> str:
        """获取风险评级"""
        if metric_type == 'drawdown':
            if value < 0.05:
                return '低'
            elif value < 0.15:
                return '中'
            else:
                return '高'
        elif metric_type == 'sharpe':
            if value > 1.5:
                return '优秀'
            elif value > 1.0:
                return '良好'
            elif value > 0.5:
                return '一般'
            else:
                return '较差'
        elif metric_type == 'volatility':
            if value < 0.2:
                return '低'
            elif value < 0.4:
                return '中'
            else:
                return '高'
        elif metric_type == 'var':
            if value > -0.05:
                return '低'
            elif value > -0.15:
                return '中'
            else:
                return '高'
        elif metric_type == 'win_rate':
            if value > 0.6:
                return '高'
            elif value > 0.4:
                return '中'
            else:
                return '低'
        return '中'
    
    def _get_position_suggestion(self, metrics: Dict) -> str:
        """获取仓位建议"""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        
        if sharpe > 1.5 and max_dd < 0.1:
            return "20-30%"
        elif sharpe > 1.0 and max_dd < 0.15:
            return "10-20%"
        elif sharpe > 0.5:
            return "5-10%"
        else:
            return "0-5%"
    
    def _get_overall_risk_level(self, metrics: Dict) -> str:
        """获取整体风险等级"""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        
        if sharpe > 1.5 and max_dd < 0.1:
            return "低风险"
        elif sharpe > 1.0 and max_dd < 0.15:
            return "中等风险"
        else:
            return "高风险"
    
    def _get_suitable_investors(self, metrics: Dict) -> str:
        """获取适合的投资者类型"""
        risk_level = self._get_overall_risk_level(metrics)
        
        if risk_level == "低风险":
            return "保守型、稳健型投资者"
        elif risk_level == "中等风险":
            return "稳健型、平衡型投资者"
        else:
            return "激进型投资者"
    
    def _get_investment_rating(self, metrics: Dict) -> str:
        """获取投资评级"""
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        
        if sharpe > 1.5 and max_dd < 0.1:
            return "强烈推荐"
        elif sharpe > 1.0 and max_dd < 0.15:
            return "推荐"
        elif sharpe > 0.5:
            return "中性"
        else:
            return "不推荐"
    
    def _get_short_term_advice(self, metrics: Dict) -> str:
        """获取短期建议"""
        return "关注技术指标变化，适时调整仓位"
    
    def _get_medium_term_advice(self, metrics: Dict) -> str:
        """获取中期建议"""
        return "结合基本面分析，制定投资计划"
    
    def _get_long_term_advice(self, metrics: Dict) -> str:
        """获取长期建议"""
        return "关注公司基本面变化，长期持有优质资产"
    
    def _get_entry_timing(self, metrics: Dict) -> str:
        """获取入场时机建议"""
        return "技术指标确认后分批入场"

# 便捷函数
def generate_analysis_report(ticker: str, analysis_data: Dict, 
                           chart_paths: Dict[str, str], 
                           metrics: Dict[str, Any]) -> str:
    """生成分析报告的便捷函数"""
    generator = PDFReportGenerator()
    return generator.generate_report(ticker, analysis_data, chart_paths, metrics)

if __name__ == "__main__":
    # 测试PDF生成
    print("PDF报告生成器模块加载完成")
    print("使用方法:")
    print("1. 创建生成器: generator = PDFReportGenerator()")
    print("2. 生成报告: generator.generate_report(ticker, data, charts, metrics)")
    print("3. 便捷函数: generate_analysis_report(ticker, data, charts, metrics)")
