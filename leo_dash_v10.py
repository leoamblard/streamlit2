# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD EXAMPLE - v3
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots


#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


#==============================================================================
# Sidebar
#==============================================================================
  

  
# Define a function to get the list of stock tickers from S&P500
@st.cache_data
def get_ticker_list():
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    return ticker_list

st.title("My Financial Dashboard ")
# Add the ticker selection on the sidebar
with st.sidebar:
    # Add a title to the sidebar section
    st.sidebar.header("Option Pannel")

    # Get the list of stock tickers from S&P500
    ticker_list = get_ticker_list()

    # Add the selection boxes
    col1, col2, col3 = st.sidebar.columns(3)  # Create 3 columns

    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = st.sidebar.selectbox("Ticker", ticker_list)

    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = st.sidebar.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", datetime.today().date())
    
    # Get company summary
    info = YFinance(ticker).info
    st.write('Company Summary')
    st.markdown('<div style="text-align: justify; font-size: 14px;">' + \
                        info['longBusinessSummary'] + \
                        '</div><br>',
                        unsafe_allow_html=True)
        
    # Get stakeholder info
    st.write("Major stakeholders")
    stake_hold = yf.Ticker(ticker).get_major_holders()
    table_html = stake_hold.to_html(index=False, header=False, classes=["no-header-table"])
    new_html = f'<div style="font-size: 14px;">{table_html}</div>'
    st.markdown(new_html, unsafe_allow_html=True)
    
    
    

#==============================================================================
# Tab 1
#==============================================================================

def render_tab1():
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """


    col1, col2= st.columns([2, 3])
    
    # Get the company information

    def  Finfo(ticker):
       
    
        return YFinance(ticker).info
    
    if ticker != '':
    
        with col1:
            info = Finfo(ticker)
            st.write(ticker,' Insights: ')
            info_keys = {'previousClose': 'Previous Close',
                         'open': 'Open',
                         'bid': 'Bid',
                         'ask': 'Ask',
                         'marketCap': 'Market Cap',
                         'volume': 'Volume'}
            stats = {}
            for key in info_keys:
                stats.update({info_keys[key]: info[key]})
    
            stats = pd.DataFrame({'Value': pd.Series(stats)})  # Convert to DataFrame
            st.dataframe(stats)

    
  
        with col2:
            df = yf.download(ticker,start= start_date,end =end_date)
            df.head()
            fig = px.line(df, x = df.index, y = df['Adj Close'], title = ticker)
            st.plotly_chart(fig)
       
#==============================================================================
# Tab 2
#==============================================================================

def render_tab2():
    
    global charter_type
    charter_type = st.selectbox('Select chart type:', ['Line', 'Candlestick'])
    
    interval = {
    '1d': '1 Day',
    '1wk': '1 Week',
    '1mo': '1 Month' }
    
    global interval_type
    interval_type = st.selectbox('Select interval:', list(interval.keys()))
    
    def ystock2(ticker, start_date, end_date, interval_type):
       stock2 = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval_type)
       df2 = stock2
       return df2
        
    if charter_type == 'Line':
        df2 = ystock2(ticker, start_date, end_date, interval_type)
        df2['MA50'] = df2['Close'].rolling(window=len(df2), min_periods=1).mean()
        
        fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        
        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2['Close'],
            name='Close Price',
            fill='tozeroy',
            fillcolor='rgba(200,230,250,0.2)',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Bar(x=df2.index,y=df2['Volume'],name='Volume',yaxis='y2',marker=dict(color='purple')))
        fig.add_trace(go.Scatter(x=df2.index, y=df2['MA50'], mode='lines', name='50-Day Moving Average'))
        
        plt.grid(False)
        
        fig.update_layout(
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=False, tickfont=dict(size=10), title='Close Price'),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=100, r=50, l=50),
        )
        
        fig.update_yaxes(title_text="Close Price", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        st.plotly_chart(fig)
        
    if charter_type == 'Candlestick':
        df2= ystock2(ticker, start_date, end_date,interval_type)

        fig = go.Figure(data=[go.Candlestick(
                open=df2['Open'],
                high=df2['High'],
                low=df2['Low'],
                close=df2['Close'])])
        
        fig.update_layout(
            xaxis=dict(showgrid=False,tickfont=dict(size=10)),  # Remove x-axis grid lines
            yaxis=dict(showgrid=False),  # Remove y-axis grid lines,
            yaxis2=dict(showgrid=False, overlaying='y', side='right', title='Bar Chart Y-Axis'),
            plot_bgcolor="rgba(0,0,0,0)",
            )
        
        st.plotly_chart(fig)
#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():

    def get_financials(stock, statement_type, period):
    # Download financial data from Yahoo Finance
        data = yf.Ticker(stock)
        
        if statement_type == 'Income Statement':
            if period == 'Annual':
                return data.financials
            elif period == 'Quarterly':
                return data.quarterly_financials
        elif statement_type == 'Balance Sheet':
            if period == 'Annual':
                return data.balance_sheet
            elif period == 'Quarterly':
                return data.quarterly_balance_sheet
        elif statement_type == 'Cash Flow':
            if period == 'Annual':
                return data.cashflow
            elif period == 'Quarterly':
                return data.quarterly_cashflow

    def main():
        
        
        stock = st.sidebar.text_input('Enter Stock Symbol:', value=ticker, key= ticker)
        statement_type = st.selectbox('Select Statement Type:', ['Income Statement', 'Balance Sheet', 'Cash Flow'])
        period = st.selectbox('Select Period:', ['Annual', 'Quarterly'])
        
        if st.button('Get Financials'):
            # Display financial data
            try:
                financial_data = get_financials(stock, statement_type, period)
                st.write(f"**{statement_type} - {period}**")
                st.write(financial_data)
                
                
                if statement_type == 'Income Statement':
                    st.line_chart(financial_data['Net Income'])
            except:
                st.error('Error: Unable to fetch financial data. Please check the stock symbol.')
    
    if __name__ == "__main__":
        main()

#==============================================================================
# Tab 4
#==============================================================================

def render_tab4():
    
    
    # Function to perform Monte Carlo simulation
    def monte_carlo_simulation(stock_data, n_simulations, time_horizon):
        returns = stock_data['Adj Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
    
        simulation_results = []
    
        for _ in range(n_simulations):
            daily_returns = np.random.normal(mean_return, std_return, time_horizon)
            price_path = np.exp(np.log(stock_data['Adj Close'].iloc[-1]) + np.cumsum(daily_returns))
            simulation_results.append(price_path)
    
        return pd.DataFrame(simulation_results).transpose()
    
    # Function to calculate VaR
    def calculate_var(simulation_results, confidence_interval=0.95):
        var = simulation_results.quantile(1 - confidence_interval, axis=1)
        return var
    
 
   
    # User input for number of simulations and time horizon
    n_simulations = st.selectbox('Number of Simulations', [200, 500, 1000])
    time_horizon = st.selectbox('Time Horizon (days)', [30, 60, 90])
    
    # Run Monte Carlo simulation
    stock_data = yf.download(ticker, start=pd.to_datetime('2020-01-01'), end=pd.to_datetime('2023-01-01'))
    simulation_results = monte_carlo_simulation(stock_data, n_simulations, time_horizon)
    
    # Calculate VaR
    var = calculate_var(simulation_results)
    
    # Plot the results
    st.line_chart(simulation_results)
    st.write('### Value at Risk:')
    st.write(f'VaR: {var.iloc[-1]:.2f}')
    
    # Optionally, you can show the last few rows of the simulation results
    st.write('### Simulation Results:')
    st.write(simulation_results.tail())

 #==============================================================================
 # Tab 5
 #==============================================================================
     

    
def render_tab5():
    #list of crypto symbols for the select box 
    symbol = [
     'BTC-USD',
     'XRP-USD',
     'TRX-USD',
     'WAVES-USD',
     'ZIL-USD',
     'ONE-USD',
     'COTI-USD',
     'SOL-USD',
     'EGLD-USD',
     'AVAX-USD',
     'NEAR-USD',
     'FIL-USD',
     'AXS-USD',
     'ROSE-USD',
     'AR-USD',
     'MBOX-USD',
     'YGG-USD',
     'BETA-USD',
     'PEOPLE-USD',
     'EOS-USD',
     'ATOM-USD',
     'FTM-USD',
     'DUSK-USD',
     'IOTX-USD',
     'OGN-USD',
     'CHR-USD',
     'MANA-USD',
     'XEM-USD',
     'SKL-USD',
     'ICP-USD',
     'FLOW-USD',
     'WAXP-USD',
     'FIDA-USD',
     'ENS-USD',
     'SPELL-USD',
     'LTC-USD',
     'IOTA-USD',
     'LINK-USD',
     'XMR-USD',
     'DASH-USD',
     'MATIC-USD',
     'ALGO-USD',
     'ANKR-USD',
     'COS-USD',
     'KEY-USD',
     'XTZ-USD',
     'REN-USD',
     'RVN-USD',
     'HBAR-USD',
     'BCH-USD',
     'COMP-USD',
     'ZEN-USD',
     'SNX-USD',
     'SXP-USD',
     'SRM-USD',
     'SAND-USD',
     'SUSHI-USD',
     'YFII-USD',
     'KSM-USD',
     'DIA-USD',
     'RUNE-USD',
     'AAVE-USD',
     '1INCH-USD',
     'ALICE-USD',
     'FARM-USD',
     'REQ-USD',
     'GALA-USD',
     'POWR-USD',
     'OMG-USD',
     'DOGE-USD',
     'SCU-SD',
     'XVS-USD',
     'ASR-USD',
     'CELO-USD',
     'RARE-USD',
     'ADX-USD',
     'CVX-USD',
     'WIN-USD',
     'C98-USD',
     'FLUX-USD',
     'ENJ-USD',
     'FUN-USD',
     'KP3R-USD',
     'ALCX-USD',
     'ETC-USD',
     'THETA-USD',
     'CVC-USD',
     'STX-USD',
     'CRV-USD',
     'MDX-USD',
     'DYDX-USD',
     'OOKI-USD',
     'CELR-USD',
     'RSR-USD',
     'ATM-USD',
     'LINA-USD',
     'POLS-USD',
     'ATA-USD',
     'RNDR-USD',
     'NEO-USD',
     'ALPHA-USD',
     'XVG-USD',
     'KLAY-USD',
     'DF-USD',
     'VOXEL-USD',
     'LSK-USD',
     'KNC-USD',
     'NMR-USD',
     'MOVR-USD',
     'PYR-USD',
     'ZEC-USD',
     'CAKE-USD',
     'HIVE-USD',
     'UNI-USD',
     'SYS-USD',
     'BNX-USD',
     'GLMR-USD',
     'LOKA-USD',
     'CTSI-USD',
     'REEF-USD',
     'AGLD-USD',
     'MC-USD',
     'ICX-USD',
     'TLM-USD',
     'MASK-USD',
     'IMX-USD',
     'XLMUSD',
     'BEL-USD',
     'HARD-USD',
     'NULS-USD',
     'TOMO-USD',
     'NKN-USD',
     'BTS-USD',
     'LTO-USD',
     'STORJ-USD',
     'ERN-USD',
     'XEC-USD',
     'ILV-USD',
     'JOE-USD',
     'SUN-USD',
     'ACH-USD',
     'TROY-USD',
     'YFI-USD',
     'CTK-USD',
     'BAN-DUSD',
     'RLC-USD',
     'TRU-USD',
     'MITH-USD',
     'AION-USD',
     'ORN-USD',
     'WRX-USD',
     'WAN-USD',
     'CHZ-USD',
     'ARPA-USD',
     'LRC-USD',
     'IRIS-USD',
     'UTK-USD',
     'QTUM-USD',
     'GTO-USD',
     'MTL-USD',
     'KAVA-USD',
     'DREP-USD',
     'OCEAN-USD',
     'UMA-USD',
     'FLM-USD',
     'UNFI-USD',
     'BADGER-USD',
     'POND-USD',
     'PERP-USD',
     'TKO-USD',
     'GTC-USD',
     'TVK-USD',
     'RAY-USD',
     'LAZIO-USD',
     'AMP-USD',
     'BICO-USD',
     'CTXC-USD',
     'FIS-USD',
     'BTG-USD',
     'TRIBE-USD',
     'QI-USD',
     'PORTO-USD',
     'DATA-USD',
     'NBS-USD',
     'EPS-USD',
     'TFUEL-USD',
     'BEAM-USD',
     'REP-USD',
     'PSG-USD',
     'WTC-USD',
     'FORTH-USD',
     'BOND-USD',
     'ZRX-USD',
     'FIRO-USD',
     'SFP-SD',
     'VTHO-USD',
     'FIO-USD',
     'PERL-USD',
     'WING-USD',
     'AKRO-USD',
     'BAKE-USD',
     'ALPACA-USD',
     'FOR-USD',
     'IDEX-USD',
     'PLA-USD',
     'VITE-USD',
     'DEGO-USD',
     'XNO-USD',
     'STMX-USD',
     'JUV-USD',
     'STRAX-USD',
     'CITY-USD',
     'JASMY-USD',
     'DEXE-USD',
     'OM-USD',
     'MKR-USD',
     'FXS-USD',
     'ETH-USD',
     'ADA-USD',
    'BNB-USD',
    'SHIB-USD']
        
    
    def get_crypto_data(symbol, start_date, end_date):
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    
    # Function to plot candlestick chart using plotly
    def plot_candlestick(data, symbol):
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
    
        fig.update_layout(title=f'{symbol} Candlestick Chart',
                          xaxis_title='Date',
                          yaxis_title='Price ($)',
                          xaxis_rangeslider_visible=False)
    
        return fig
    

    
    # Select cryptocurrency symbol and date range
    symbol = st.selectbox('Enter cryptocurrency symbol (e.g., BTC-USD):', symbol)
    
    # Fetch cryptocurrency data
    crypto_data = get_crypto_data(symbol, start_date, end_date)
    
    # Display candlestick chart
    if not crypto_data.empty:
        st.plotly_chart(plot_candlestick(crypto_data, symbol))
    else:
        st.warning('No data available for the selected date range.')
    

    




# Render the tabs
tab1, tab2, tab3, tab4,tab5, = st.tabs(["Summary","Chart", "Financials","Monte Carlo", "Crypto"])


#Render tabs
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()




