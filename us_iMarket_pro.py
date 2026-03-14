import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import feedparser
import urllib.parse
from datetime import datetime
import google.generativeai as genai # 添加这一行
import numpy as np
st.markdown("""
    <style>
    /* 1. 核心修复：对中英文通用的自动换行 */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        /* white-space: pre-wrap 有时会保留不必要的空格导致英文排版难看 */
        /* 改用 normal 配合 break-word 是中英文混排的最佳实践 */
        white-space: normal !important; 
        word-wrap: break-word !important;
        overflow-wrap: anywhere !important;
        line-height: 1.6 !important;
        letter-spacing: normal !important;
    }

    /* 2. 针对中文的特殊优化：允许中文字符间断行 */
    /* 针对英文的特殊优化：防止长单词撑破容器 */
    [data-testid="stMarkdownContainer"] {
        text-align: justify;
        text-justify: inter-word;
    }

    /* 3. 修复你截图 1 中看到的“字母散开”问题 */
    /* 强制重置可能被误写的 letter-spacing */
    .stMarkdown * {
        letter-spacing: 0px !important;
    }
    
    /* 4. 针对估值报告容器 */
    div[data-testid="stNotification"] {
        word-break: break-word !important;
    }
    </style>
""", unsafe_allow_html=True)
# --- 1. Basic Configuration ---
st.set_page_config(
    page_title="iMarket AI Assistant: Smart Decision Engine", 
    page_icon="🤖",
    layout="wide"
    )
# --- 新增：稳健型价格抓取函数 (防止 $nan) ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 1. 优先从 info 获取
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose')

        # 2. 如果 info 拿不到数据（常发生于盘后或 API 限制）
        if current_price is None or (isinstance(current_price, float) and np.isnan(current_price)):
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            else:
                current_price, prev_close = 0.0, 0.0
        
        return float(current_price), float(prev_close if prev_close else current_price)
    except:
        return 0.0, 0.0
    
# --- 新增：深度估值计算函数 ---
def get_advanced_valuation(ticker, discount_rate=0.15):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. 基础数据提取 (yfinance 默认返回完整数值)
        fcf = info.get('freeCashflow') or info.get('operatingCashflow', 0) * 0.8 # 备选方案
        shares = info.get('sharesOutstanding', 0)
        curr_price = info.get('currentPrice', 1)
        
        # 2. 净现金调整 (Net Cash = Total Cash - Total Debt)
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        net_cash = total_cash - total_debt
        
        if fcf <= 0 or shares <= 0:
            return None

        # 3. 保守型 DCF 计算
        growth_rate = 0.05  # 前5年增长率
        perp_growth = 0.02  # 永续增长率
        
        # 计算前5年现值
        pv_fcf = 0
        for i in range(1, 6):
            future_fcf = fcf * (1 + growth_rate)**i
            pv_fcf += future_fcf / (1 + discount_rate)**i
        
        # 计算终值 (Terminal Value) 并折现
        terminal_v = (fcf * (1 + growth_rate)**5 * (1 + perp_growth)) / (discount_rate - perp_growth)
        pv_tv = terminal_v / (1 + discount_rate)**5
        
        # 4. 企业价值转股权价值 (加上净现金)
        # 内在价值 = (经营价值 + 净现金) / 总股本
        dcf_intrinsic_value = (pv_fcf + pv_tv + net_cash) / shares
        
        # 防止极端负值（如果债务远超现金和现金流现值）
        dcf_intrinsic_value = max(dcf_intrinsic_value, 0)
        
        upside = (dcf_intrinsic_value / curr_price - 1) * 100

        return {
            "dcf_price": dcf_intrinsic_value,
            "upside_pct": upside,
            "ev_sales": info.get('enterpriseToRevenue', 0),
            "ev_gp": info.get('enterpriseValue', 0) / info.get('grossProfits', 1) if info.get('grossProfits') else 0,
            "sector": info.get('sector', 'N/A')
        }
    except Exception as e:
        print(f"Valuation Error: {e}")
        return None

# --- 新增：模型专用 AI 分析函数 ---
def run_valuation_model_analysis(ticker, val_data, lang="中文"):
    """
    执行 iMarket Pro 深度估值诊断。
    基于 15% 严苛折现率的 DCF 模型进行量化判读。
    """
    # 1. 密钥安全检查
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ 错误：未在 Streamlit Cloud 后台配置 GEMINI_API_KEY"
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
 
    try:
        # 2. 动态获取可用模型 (避开 404 错误)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = next((m for m in available_models if 'flash' in m.lower()), None)
        
        if not target_model:
            target_model = available_models[0] if available_models else "gemini-1.5-flash"
            
        model = genai.GenerativeModel(target_model)

        # 3. 构建硬核量化 Prompt (方案 B)
        if lang == "English":
            opening = f"**iMarket AI Valuation Engine:** Analysis for **{ticker}** finalized. Model penetration complete."
            prompt = f"""
            Role: Senior Quantitative Strategist & Valuation Expert.
            Task: Evaluate {ticker} using a conservative 15% discount rate DCF model.
            
            Input Quant Data:
            - Intrinsic Value (DCF): ${val_data['dcf_price']:.2f}
            - Current Upside: {val_data['upside_pct']:.1f}%
            - EV/Sales: {val_data['ev_sales']:.2f}x
            - EV/GP (Moat Strength): {val_data['ev_gp']:.2f}x
            
            Reporting Requirements:
            1. Start with: "{opening}"
            2. Analyze the 'Margin of Safety' based on our 15% discount hurdle.
            3. Differentiate between a 'Golden Pit' and a 'Value Trap'.
            4. Use CLEAR bullet points with double spacing.
            5. NO complex LaTeX math blocks. Use bold numbers only.
            """
        else:
            opening = f"**iMarket AI 估值引擎：** **{ticker}** 建模分析已穿透。模型测算逻辑已就绪。"
            prompt = f"""
            角色：顶级华尔街量化策略师。
            任务：基于 15% 严苛折现率的 DCF 模型，对 {ticker} 进行内在价值判读。
            
            底层建模数据：
            - DCF 内在价值: ${val_data['dcf_price']:.2f}
            - 预期空间: {val_data['upside_pct']:.1f}%
            - EV/Sales (规模定价): {val_data['ev_sales']:.2f}x
            - EV/GP (护城河强度): {val_data['ev_gp']:.2f}x
            
            报告要求：
            1. 开场白必须是："{opening}"
            2. 基于 15% 的高贴现率标准，深度辨析其“安全边际”与“估值陷阱”风险。
            3. 每个要点之间必须空两行，严禁文字堆叠。
            4. 严禁生成 LaTeX 公式，数字与汉字间加空格。
            5. 结论需明确：是“利空出尽的黄金坑”还是“逻辑崩坏的陷阱”。
            """

        # 4. 生成响应并处理格式
        response = model.generate_content(prompt)
        # 强制增加换行符，确保 Markdown 渲染清晰
        clean_text = response.text.replace("\n*", "\n\n*").replace("\n-", "\n\n-")
        return clean_text

    except Exception as e:
        # 捕获异常，防止整个 Streamlit 应用崩溃
        return f"❌ AI 诊断引擎暂时不可用: {str(e)}"

def run_gemini_pro_analysis(ticker, tech_metrics, news_summary, language="中文"):
    # 1. 密钥检查
    if "GEMINI_API_KEY" in st.secrets:
        api_key_val = st.secrets["GEMINI_API_KEY"]
    else:
        return "❌ 错误：未在 Streamlit Cloud 后台配置 GEMINI_API_KEY"
    
    genai.configure(api_key=api_key_val)
    
    try:
        # 2. 自动获取可用的模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            return "❌ 你的 API Key 没有任何可用模型。"
        
        # 优先使用 flash 模型以获得极速响应
        target_model = next((m for m in available_models if 'flash' in m.lower()), available_models[0])
        model = genai.GenerativeModel(target_model)

        # 3. 核心逻辑：根据语言定制 Prompt (整合了你最新的润色内容)
# 3. 核心逻辑：根据语言定制 Prompt
        # 获取实时价格（从传入的指标中显式提取，或外部传入）
        price_info = tech_metrics.get('current_price', 'N/A') if isinstance(tech_metrics, dict) else "N/A"

        if language == "English":
            role_inv = "iMarket AI Assistant (Powered by Gemini 1.5)"
            prompt = f"""
            Role: You are {role_inv}, a sophisticated investment co-pilot.
            Task: Generate an intelligence report for {ticker}.
            
            [CRITICAL DATA]: 
            - Current Real-time Price: {price_info} 
            - Technical Context: {tech_metrics}
            - News Sentiment: {news_summary}
            
            ... (rest of the prompt)
            """
        else:
            role_inv = "iMarket AI 智能助手 (由 Gemini 1.5 驱动)"
            prompt = f"""
            角色：你是 {role_inv}，一位专业的投资副驾。
            任务：为用户生成针对 {ticker} 的全维度智能诊断报告。
            
            【实时监控数据】：
            - 标的代码：{ticker}
            - 实时价格：{price_info} 
            - 技术指标：{tech_metrics}
            - 市场情绪：{news_summary}
            
            请按以下专业框架输出：
            1. 🤖 **助手速评 (Insights)**：基于当前价格 {price_info} 点睛核心局势。
            ... (其余框架部分保持不变)
            """

        # 4. 执行生成并返回
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        # 捕获所有运行时的 API 或逻辑错误
        return f"❌ AI 分析出错: {str(e)}"
    






# --- 2. Top Market Indices (Color-Coded) ---

@st.cache_data(ttl=300)
def fetch_market_indices():
    indices = {
        "DJIA": "^DJI", "NDX": "^NDX", "SPX": "^GSPC",
        "TSX": "^GSPTSE", "Crude": "CL=F", "Gold": "GC=F", 
        "USDX": "DX=F"  # <--- 将 DX-Y.NYB 改为 DX=F (美元指数期货)
    }
    # ... 其余逻辑不变 ...
    try:
        # 1. 下载原始数据
        data = yf.download(list(indices.values()), period="2d", interval="1d", auto_adjust=True)
        
        # --- 📍 替换开始：将原来的 close_data = ... 替换为以下逻辑 ---
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                close_data = data['Close']
            else:
                close_data = data 
        else:
            close_data = data
        # --- 📍 替换结束 ---

        results = {}
        
        for name, sym in indices.items():
            # 确保 sym 存在于 columns 中
            if sym in close_data.columns:
                series = close_data[sym].dropna()
                if len(series) >= 2:
                    curr, prev = series.iloc[-1], series.iloc[-2]
                    diff = curr - prev 
                    pct = (diff / prev) * 100 
                    results[name] = {"val": curr, "diff": diff, "pct": pct}          
        return results
    
    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return {}

# --- 3. Main Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_financial_data(ticker, days):
    try:
        data = yf.download([ticker, "^VIX"], period=f"{days}d", interval="1d", auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        else:
            prices = data[['Adj Close']]
        return prices
    except:
        return pd.DataFrame()

def get_reddit_sentiment(ticker):
    """
    模拟散户热度指标：结合成交量波动与价格乖离度
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty: return 0, "Neutral"
        
        # 计算今日成交量与 5 日均值的比率
        avg_vol = hist['Volume'].mean()
        curr_vol = hist['Volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol
        
        # 逻辑：成交量翻倍通常意味着社交媒体热度飙升
        mentions = int(vol_ratio * 10) # 模拟提及次数
        
        if vol_ratio > 2.0:
            score = "High Heat 🔥"
        elif vol_ratio > 1.2:
            score = "Increasing"
        else:
            score = "Quiet"
            
        return mentions, score
    except:
        return 0, "N/A"

# --- 4. Sidebar Control ---
st.sidebar.image("iMarket Pro.png", use_container_width=True)
st.sidebar.title("Control Center")
ticker_input = st.sidebar.text_input("Ticker (e.g., AAPL 🇺🇸 | AC.TO 🇨🇦)", "AAPL").upper()

if ".TO" in ticker_input or ".V" in ticker_input:
    st.sidebar.success("Canadian Ticker Detected 🇨🇦")
    ticker = ticker_input
else:
    ticker = ticker_input
    if len(ticker_input) >= 2 and ticker_input.isalpha():
        st.sidebar.info("💡 Tip: For Canada, add .TO")

lookback = st.sidebar.slider("Lookback Period (Divergence)", 30, 250, 90)


st.sidebar.markdown("---")
# 这里的变量名 report_lang 必须对应你 352 行使用的名字
report_lang = st.sidebar.selectbox(" 🌐🇺🇸/🇨🇳", ["English", "中文"])


st.sidebar.markdown("---")
st.sidebar.caption("🚀  Designed by J")
st.sidebar.caption("🤖  Powered by Gemini AI")
st.sidebar.caption("📅  v3.0 | March 2026")


# --- 5. Market Index Bar Execution ---


# --- 品牌标题：位置上移并微调黑字副标题 ---
st.markdown(
    """
    <div style="text-align: center; margin-top: -40px; margin-bottom: 5px; padding-top: 0px;">
        <h1 style="
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            font-weight: 800; 
            background: linear-gradient(135deg, #d4af37 25%, #f7e7ce 50%, #d4af37 75%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            text-shadow: 1px 1px 8px rgba(212, 175, 55, 0.2);
            letter-spacing: -1px; 
            margin-bottom: 0px;
            font-size: 3.2rem;
            line-height: 1.1;
        ">
            iMarket Pro
        </h1>
        <p style="
            color: #000000; 
            font-size: 1.1rem; 
            font-weight: 600; 
            letter-spacing: 2px; 
            text-transform: uppercase;
            margin-top: -8px;
            margin-bottom: 10px;
        ">
            AI-Powered Market Research Engine
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# 1. 调用抓取函数
index_data = fetch_market_indices()

# 2. 只有当抓取到数据时才渲染
if index_data:
    # 关键：使用 len(index_data) 动态分列
    # 这样当你字典里有 7 个指数（含 USDX）时，它会自动创建 7 列
    idx_cols = st.columns(len(index_data))
    
    for i, (name, d) in enumerate(index_data.items()):
        # 格式化 delta 字符串
        delta_str = f"{d['diff']:+.2f} ({abs(d['pct']):.2f}%)"
        
        # 渲染到对应的列中
        idx_cols[i].metric(
            label=name, 
            value=f"{d['val']:,.2f}", 
            delta=delta_str, 
            delta_color="normal"
        )
st.divider()
st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); 
        padding: 30px; 
        border-radius: 20px; 
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    ">
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 2.5rem;">🤖</span>
            <div>
                <h1 style="margin: 0; color: #ffffff; font-size: 2.2rem; letter-spacing: -0.5px;">
                    iMarket AI Assistant <span style="color: #60a5fa; font-weight: 300;">| {ticker_input}</span>
                </h1>
                <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem;">
                    Smart Decision Engine • Technical Insights • Deep Valuation
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
# --- 6. Main Indicators & Charts ---
prices = fetch_financial_data(ticker, lookback)

if not prices.empty and ticker in prices.columns:
    # Indicator Logic
    delta = prices[ticker].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    
    current_vix = prices["^VIX"].iloc[-1] if "^VIX" in prices.columns else 0
    vix_sma = prices["^VIX"].rolling(20).mean().iloc[-1] if "^VIX" in prices.columns else 1

    # Metrics Section
    
    # --- 修改处：看板数据调用 ---
    price_val, prev_val = get_stock_data(ticker)
    st.subheader(f"⚠️ {ticker} Real-time Sentiment Warning")
    mentions, wsb_score = get_reddit_sentiment(ticker)
    m1, m2, m3, m4 = st.columns(4)
    # 动态显示涨跌幅
    price_delta = price_val - prev_val
    
    # --- 价格与涨跌逻辑计算 ---
    if price_val > 0:
        # 1. 计算绝对值变化 (例如 -5.05)
        change_abs = price_val - prev_val
        
        # 2. 计算百分比变化 (例如 -1.97%)
        change_pct = (change_abs / prev_val) * 100 if prev_val != 0 else 0
        
        # 3. 组合成你想要的格式: "-5.05 (-1.97%)"
        # +符号会自动处理正负号，.2f 保留两位小数
        delta_display = f"{change_abs:+.2f} ({change_pct:+.2f}%)"

        # --- 渲染到界面 ---
        m1.metric(
            label="Price", 
            value=f"${price_val:.2f}", 
            delta=delta_display,
            delta_color="normal" # 自动：正数绿，负数红
        )
    else:
        m1.metric("Price", "Data Error", delta=None)
    

    m2.metric("RSI", f"{rsi_series.iloc[-1]:.2f}", delta="OB" if rsi_series.iloc[-1] > 70 else "OS" if rsi_series.iloc[-1] < 30 else "Normal")
    m3.metric("VIX", f"{current_vix:.2f}", delta=f"{((current_vix/vix_sma)-1)*100:.1f}%", delta_color="inverse")
    m4.metric("WSB", f"{mentions}", delta="Sentiment Check")

    # Technical Chart
    st.subheader("📈 Technical Analysis (Bollinger + MACD)")
    daily = yf.download(ticker, period="1y", interval="1d")
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.droplevel(1)
    
    ma20 = daily['Close'].rolling(20).mean()
    std20 = daily['Close'].rolling(20).std()
    up_bb, lo_bb = ma20 + (std20 * 2), ma20 - (std20 * 2)
    macd = daily['Close'].ewm(span=12).mean() - daily['Close'].ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    hist = macd - sig

    apds = [
        mpf.make_addplot(up_bb, color='gray', alpha=0.2),
        mpf.make_addplot(lo_bb, color='gray', alpha=0.2),
        mpf.make_addplot(macd, panel=2, color='fuchsia', ylabel='MACD'),
        mpf.make_addplot(sig, panel=2, color='blue'),
        mpf.make_addplot(hist, panel=2, type='bar', color='gray', alpha=0.3)
    ]
    fig, axlist = mpf.plot(daily, type='candle', style='yahoo', volume=True, mav=(20, 50, 200), addplot=apds, panel_ratios=(6,2,2), returnfig=True, figsize=(12, 8))
    axlist[0].legend(['MA20', 'MA50', 'MA200', 'Upper BB', 'Lower BB'], loc='upper left', fontsize='x-small')
    st.pyplot(fig)

    # Divergence Chart
    st.divider()
    st.subheader("🔍 Price Momentum & Technical Divergence")
    fig_div, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax_p.plot(prices.index, prices[ticker], color='#1f77b4', label="Price")
    ax_r.plot(rsi_series.index, rsi_series, color='#9467bd', label="RSI")
    ax_r.axhline(70, color='red', ls='--'); ax_r.axhline(30, color='green', ls='--')
    ax_p.legend(); ax_r.legend()
    st.pyplot(fig_div)
    
    
# --- 8. Chart Legend Expander (Professional Analysis) ---
    if report_lang == "English":
        with st.expander("📖 Professional Analysis: RSI & Volume Divergence"):
            st.markdown(f"""
            ### 1. RSI Divergence: Momentum Exhaustion
            RSI measures the 'speed' and 'strength' of price movements.

            #### **A. Bearish Divergence —— Exit Signal**
            * **Phenomenon**: Price hits a **new high**, but RSI line is trending **downward** (lower peak).
            * **Meaning**: Upward momentum is fading despite rising prices. Like a car sprinting on an empty tank.
            * **Action**: Consider reducing positions or raising stop-loss levels.

            #### **B. Bullish Divergence —— Buy Signal**
            * **Phenomenon**: Price hits a **new low**, but RSI line is trending **upward** (higher trough).
            * **Meaning**: Selling pressure is exhausting. 
            * **Action**: System detects **{ticker}** may be in this zone; a rebound is often imminent.

            ---

            ### 2. Volume Divergence: Capital Support
            Volume is the "fuel" of a stock. **Rising price with rising volume** is the healthiest trend.

            #### **A. Low Volume Rally —— False Prosperity**
            * **Meaning**: Buying power is depleted; usually retail chasing while institutions exit. High risk of sharp reversal.

            #### **B. High Volume Crash —— Panic Selling**
            * **Meaning**: Massive panic selling. If at the end of a downtrend, it signals a "washout"; if at a peak, it's a disaster.

            #### **C. Low Volume Pullback —— Consolidation**
            * **Meaning**: Selling is not aggressive; usually healthy profit-taking or institutional "shaking the tree".
            """)
    else:
        with st.expander("📖 核心技术指标深度解读：RSI 与 量价背离"):
            st.markdown(f"""
            ### 1. RSI 背离：判断“动力”是否衰竭
            RSI 衡量的是价格上涨或下跌的“速度”和“力度”。

            #### **A. 看跌背离 (Bearish Divergence) —— 逃顶信号**
            * **现象**：股价创出**新高**，但 RSI 线却在走**下坡路**（高点比前一个高点低）。
            * **含义**：虽然价格在涨，但支撑上涨的动能正在减弱。
            * **操作**：建议减仓或调高止损位。

            #### **B. 看涨背离 (Bullish Divergence) —— 抄底信号**
            * **现象**：股价创出**新低**，但 RSI 线却在走**上坡路**（低点比前一个低点高）。
            * **含义**：下跌的杀伤力已经减弱，空头力量正在衰竭。
            * **操作**：系统检测到 **{ticker}** 可能正处于此类信号中。
            """)

    # --- 9. Technical Indicators: Overbought, Oversold & Overextended ---
    if report_lang == "English":
        with st.expander("💡 Pro Guide: Identifying Overbought vs. Oversold"):
            st.markdown(f"""
            ### 1. Overbought —— Warning of Pullback
            * **Technical**: **RSI > 70** or price touching the **Upper Bollinger Band**.
            * **Strategy**: Take profit or reduce exposure; avoid chasing highs.

            ### 2. Oversold —— Watching for Rebound
            * **Technical**: **RSI < 30** or price piercing the **Lower Bollinger Band**.
            * **Strategy**: Potential buying opportunity; look for volume confirmation.

            ### 3. Overextended (Deep Value) —— Finding the Limit
            * **Definition**: Deeper than oversold; price is significantly below the 200-day MA.
            * **Technical**: **RSI < 20** and extreme negative Bias.
            * **Strategy**: High risk-reward ratio for "revenge rebounds."

            ---

            ### ⚠️ Professional Tips
            1. **Trend Trap**: In strong trends, RSI can stay overbought/oversold for a long time. 
            2. **Double Confirmation**: The signal is strongest when RSI crosses back inside the 30/70 levels.
            3. **Context**: If **VIX** is rising while **{ticker}** is oversold, the rebound probability increases.
            """)
    else:
        with st.expander("💡 进阶指南：如何识别超买、超卖与超跌"):
            st.markdown(f"""
            ### 1. 超买 (Overbought) —— 警惕回调
            * **技术识别**：**RSI > 70** 或股价触碰**布林带上轨**。
            * **操作策略**：通常是减仓信号，不建议此时追涨。

            ### 2. 超卖 (Oversold) —— 关注反弹
            * **技术识别**：**RSI < 30** 或股价穿出**布林带下轨**。
            * **操作策略**：潜在买入机会，需配合成交量确认。

            ### 3. 超跌 (Overextended) —— 寻找极限
            * **核心区别**：比超卖更严重，股价远低于 MA200 均线。
            * **操作策略**：极易引发“报复性反弹”。

            ---

            ### ⚠️ 交易员笔记 (Professional Tips)
            1. **趋势陷阱**：超买不代表立刻跌，超卖不代表立刻涨。
            2. **双重确认**：最可靠信号是 RSI 回到正常区间内。
            3. **结合背景**：系统检测 **{ticker}** 指标时，请同步关注 VIX 指数。
            """)
            
    # VIX & Earnings
    st.divider()
    vix_col, earn_col = st.columns([2, 1])
    with vix_col:
        st.subheader("📉 VIX Volatility Trend")
        vix_df = yf.download("^VIX", period=f"{lookback}d")
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)
        fig_v, ax_v = plt.subplots(figsize=(8, 3))
        ax_v.plot(vix_df.index, vix_df['Close'], color='red')
        ax_v.axhline(20, color='orange', ls='--')
        ax_v.fill_between(vix_df.index, vix_df['Close'], 20, where=(vix_df['Close'] > 20), color='red', alpha=0.1)
        st.pyplot(fig_v)

# --- Integrated Tested Earnings Module (V3 Refined) ---
    with earn_col:
        ticker_obj = yf.Ticker(ticker)
        next_earn_date = None
        
        try:
            # 1. 优先使用新版接口
            earnings = ticker_obj.get_earnings_dates(limit=1)
            if earnings is not None and not earnings.empty:
                next_earn_date = earnings.index[0].date()
            
            # 2. 如果失败，尝试备选 calendar 接口
            if next_earn_date is None:
                cal = ticker_obj.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    next_earn_date = cal.get('Earnings Date')[0].date()
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    next_earn_date = cal.iloc[0, 0].date()
        except:
            pass # 失败时不报错，静默处理

        # --- 核心改进：只有拿到日期才显示 UI ---
        if next_earn_date:
            st.subheader("📅 Earnings Schedule") # 只有成功了才显示标题
            days_left = (next_earn_date - datetime.now().date()).days
            
            if days_left >= 0:
                st.info(f"Next Earnings: **{next_earn_date}** (In **{days_left}** days)")
                if 0 <= days_left <= 7:
                    st.error("⚠️ Earnings Week: High IV expected!")
            else:
                # 日期已过（比如是昨天），显示为“待定”
                st.caption("📅 Next Earnings: TBA (Post-Earnings Period)")
        else:
            # 自动隐藏警告：如果获取不到，直接不占地方，或者只留一个极小的灰色提示
            # 删掉原本的 st.warning，改为下面这种不显眼的提示
            st.caption("📅 Earnings info currently unavailable from Yahoo Finance")
            
        
        
        

    # --- Integrated Tested News Module ---
    st.divider()
    st.subheader(f"📰 {ticker} English Market News")

    @st.cache_data(ttl=600)
    def fetch_2026_news(symbol):
        news_items = []
        try:
            raw_yf = yf.Ticker(symbol).news
            for item in raw_yf[:5]:
                title = item.get('title') or item.get('headline') or (item.get('content', {}).get('title')) or "News Update"
                link = item.get('link') or item.get('url') or "https://finance.yahoo.com"
                ts = item.get('providerPublishTime') or item.get('pubDate')
                p_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if isinstance(ts, int) else "Recently"
                news_items.append({'title': title, 'link': link, 'source': item.get('publisher') or "Yahoo", 'time': p_time})
        except: pass

        try:
            safe_q = urllib.parse.quote(f"{symbol} stock")
            rss_url = f"https://google.com{safe_q}&hl=en-US&gl=US"
            feed = feedparser.parse(rss_url)
            for e in feed.entries[:5]:
                news_items.append({'title': e.title, 'link': e.link, 'source': getattr(e, 'source', {}).get('title', 'Google News'), 'time': e.published})
        except: pass
        return news_items

    final_news = fetch_2026_news(ticker)
    if final_news:
        for item in final_news:
            with st.container():
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['source']} | {item['time']}")
                st.write("---")
    else:
        st.error("❌ Failed to retrieve news. Run: `pip install -U yfinance`.")


    # --- 10. Gemini AI 深度决策系统 (双引擎版) ---
    st.divider()

    # 【1. 数据准备区】：确保所有按钮都能拿到最新的数据
    current_rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else "N/A"
    tech_data = {
        "price": f"${prices[ticker].iloc[-1]:.2f}",
        "rsi": f"{current_rsi_val:.2f}",
        "vix": f"{current_vix:.2f}",
        "lookback": f"{lookback} days"
    }
    news_titles = [item['title'] for item in final_news] if final_news else "No recent news found."

    # 【2. 动态 UI 文字配置】
    if report_lang == "English":
        h_text = "🤖 AI Decision Dual Engines"
        b1_text = "🚀 Real-time AI Report"
        b2_text = "💎 Deep Valuation Model"
        s1_text = "Analyzing Technicals & News..."
        s2_text = "Running DCF & Valuation Models..."
    else:
        h_text = "🤖 AI 决策双引擎系统"
        b1_text = "🚀 生成实时 AI 报告"
        b2_text = "💎 运行深度估值模型"
        s1_text = "正在联网分析技术面与新闻..."
        s2_text = "正在运行 DCF 与行业对比模型..."

    st.header(h_text)

    # 【3. 按钮布局区】：并列展示
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
            # 按钮 1：增加价格注入逻辑
            if st.button(b1_text, width='stretch'): # 顺便修复 width 警告
                with st.spinner(s1_text):
                    # 1. 在调用前，确保拿到当前 UI 显示的价格
                    curr_p, _ = get_stock_data(ticker)
                    
                    # 2. 将价格手动加入 tech_data 字典（如果它是个字典）
                    # 如果 tech_data 只是个字符串，就直接在调用时拼进去
                    if isinstance(tech_data, dict):
                        tech_data['current_price'] = curr_p
                    
                    # 3. 调用函数
                    report = run_gemini_pro_analysis(ticker, tech_data, news_titles, report_lang)
                    st.markdown(report)


    with col_btn2:
        if st.button(b2_text, use_container_width=True):
            with st.spinner(s2_text):
                # 1. 增加重试逻辑，确保不是因为网络波动导致失败
                v_data = get_advanced_valuation(ticker, 0.15)
                
                # 如果第一次失败，尝试清除缓存并重抓一次
                if not v_data:
                    yf.Ticker(ticker).history(period="1d") # 激活连接
                    v_data = get_advanced_valuation(ticker, 0.15)

                # 2. 检查数据并渲染
                if v_data and v_data.get('dcf_price') is not None:
                    # 展现核心数据卡片
                    m1, m2, m3 = st.columns(3)
                    
                    # 标定估值颜色：upside > 0 绿色，upside < 0 红色
                    u_color = "normal" if v_data['upside_pct'] > 0 else "inverse"
                    
                    m1.metric("DCF Value", f"${v_data['dcf_price']:.2f}", f"{v_data['upside_pct']:+.1f}%", delta_color=u_color)
                    m2.metric("EV/Sales", f"{v_data['ev_sales']:.2f}x")
                    m3.metric("EV/GP", f"{v_data['ev_gp']:.2f}x")
                    
                    # 调用 AI 分析
                    model_report = run_valuation_model_analysis(ticker, v_data, report_lang)
                    st.info(model_report)
                else:
                    # 3. 增强型错误提示，方便你排查
                    st.error(f"⚠️ Financial data unavailable for {ticker}. This might be due to Yahoo Finance API limits on cloud servers. Please try again in 10 seconds.")


    with st.expander("📖 核心估值模型深度解读：DCF 与 企业价值倍数" if report_lang=="中文" else "📖 Deep Dive: DCF & Valuation Multiples"):
        if report_lang == "中文":
            st.markdown("""
            ### 1. DCF (贴现现金流) - 寻找内在价值
            * **原理**：DCF 认为公司现在的价值等于它未来能赚到的所有钱“折现”到今天的总和。
            * **高折现率策略**：本系统默认采用 **15% 折现率**。这是一个极度保守的“滤网”，只有当股价远低于这个标准时，才具有真正的**安全边际**。
            
            ### 2. EV/Sales (企业价值/销售额) - 规模与定价权
            * **逻辑**：相比 P/S，EV 考虑了公司的负债。
            * **判读**：如果该指标显著低于行业平均，可能存在**低估**；如果极高且缺乏增长支撑，则是**估值泡沫**。

            ### 3. EV/Gross Profit (企业价值/毛利) - 护城河指标
            * **核心**：这是衡量 AI 与软件公司最硬核的指标。它反映了公司每 1 元毛利在市场上被赋予的溢价。
            * **百分位意义**：查看当前倍数在过去 5 年的位置。处于 **20% 分位以下** 通常意味着处于“历史性底部”。
            """)
        else:
            st.markdown("""
            ### 1. DCF (Discounted Cash Flow) - The Intrinsic Value
            * **Principle**: DCF posits that a company is worth the sum of all its future cash flows, brought back to present value.
            * **High Discount Rate**: We use a **15% Discount Rate** by default. This acts as a conservative filter, ensuring a significant **Margin of Safety**.
            
            ### 2. EV/Sales - Scale & Pricing Power
            * **Logic**: Unlike P/S, EV (Enterprise Value) accounts for the company's debt and cash levels.
            * **Interpretation**: Significantly lower than industry average suggests **undervaluation**; excessively high suggests a **valuation bubble**.

            ### 3. EV/Gross Profit - The Moat Metric
            * **Core**: The ultimate metric for AI & SaaS firms. It shows the premium the market pays for every $1 of gross profit.
            * **Percentile**: Metrics below the **20th percentile** over 5 years often indicate a "Historical Floor."
            """)
    with st.expander("💡 进阶指南：如何区分“黄金坑”与“估值陷阱”" if report_lang=="中文" else "💡 Advanced Guide: Golden Pit vs. Value Trap"):
        if report_lang == "中文":
            st.info("""
            **🔍 识别黄金坑 (Golden Pit)**
            - **指标**：DCF 空间 > 20% 且 EV/GP 处于历史低位。
            - **信号**：AI 报告中提到“利空出尽”、“基本面改善”或“机构暗中吸筹”。
            
            **⚠️ 警惕估值陷阱 (Value Trap)**
            - **指标**：估值看起来极低，但 DCF 计算显示未来现金流正在萎缩。
            - **信号**：新闻中频繁出现“裁员”、“核心技术流失”或“法律诉讼”。
            """)
        else:
            st.info("""
            **🔍 Identifying a Golden Pit**
            - **Metrics**: DCF Upside > 20% and EV/GP at historical lows.
            - **Signals**: AI report mentions "Negative news priced in" or "Fundamental turnaround."
            
            **⚠️ Beware of Value Traps**
            - **Metrics**: Ratios look cheap, but DCF reveals shrinking future cash flows.
            - **Signals**: Frequent news regarding "Layoffs," "Loss of key talent," or "Litigation."
            """)
            
else:
    st.error("❌ Data Fetch Failed. Check connection or Ticker.")






