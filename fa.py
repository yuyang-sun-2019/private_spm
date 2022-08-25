from matplotlib.colors import DivergingNorm
import streamlit as st
from bokeh.models.widgets import Div

from pandas_datareader import data
import matplotlib.pyplot as plt
import plotly


from scipy.stats import norm

import pyfolio as pf

from datetime import datetime
from dateutil.relativedelta import relativedelta


import pickle
import altair as alt
from altair.expr import datum
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# %matplotlib inline

#pickle_in = open('classifier2.pkl', 'rb') 
#classifier = pickle.load(pickle_in)

st.sidebar.title("One Stop Solution")
option = st.sidebar.selectbox("Service Selection",('Select a service','See Trends','Sentiment Analysis'))

#financial_data

panel_data = pd.read_csv("stockdata1.csv",header=[0,1,2])
close = panel_data[['Close','Adj Close']]


#Annualized Volatility
annualized_hv = np.sqrt(np.log(close['Close']/close['Close'].shift(1)).var()) * np.sqrt(252)
ah=pd.DataFrame([annualized_hv]).T
ah= ah.reset_index(level=0)
ah= ah.set_axis(['Symbol', 'Value'], axis=1, inplace=False)

sentiment=pd.read_csv("plot.csv")






df2=pd.read_csv("stockdata2.csv")
#df2.drop(['Unnamed: 0'],axis=1,inplace=True)
tweets=pd.read_csv("finaldataset.csv")


if option == 'Select a service':
    img = Image.open('./flt.jpeg')
    st.image(img, width=700)
    original_title = '<p style="background-color: antiquewhite ; font-family:Courier; text-align:center; color:black; font-size: 40px;">ANALYSIS OF THE STOCK MARKET</p>'
    st.markdown(original_title, unsafe_allow_html=True) 
    original_title = '<p style="font-family:Courier; text-align:center; color:Black; font-size: 25px;">GOING BEYOND JUST THE NUMBERS</p>'
    st.markdown(original_title, unsafe_allow_html=True) 

if option == 'See Trends':
    #img3 = Image.open('./trends.jpeg')
    #st.image(img3, width=700)
    original_title = '<p style="background-color:White; font-family:Courier; text-align:center; color:Black; font-size: 30px;">SEE BASIC TRENDS</p>'
    st.markdown(original_title, unsafe_allow_html=True) 
    original_title = '<p style="font-family:Courier; text-align:center; color:White; font-size: 25px;">Fundamental Analysis of the stocks at YOUR convenience</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    

    

    visualtype = st.sidebar.selectbox('Select the visualisation you want to see',('Select a visualisation','General trends','Annualized Volatility','Return Distribution','Moving Averages','Correlation Plots',))

    if visualtype == 'General trends':

        #Inputs
        stock = st.sidebar.multiselect('Select any of the stocks for Analysis',list(tweets.Company.unique()), ['AAPL','MSFT'])
        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

        option = st.sidebar.selectbox('What do you want to compare?',('Volume traded','Close Prices','Adjusted Close Prices'))

        #Calcs with inputs
        mask_stock = df2['Company'].isin(stock)
        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')

        df2_filtered= df2[mask_stock]
        df3=df2_filtered[(df2_filtered['Date'] > mask_start_time) & (df2_filtered['Date'] < mask_end_time)]
        panel_data_filtered=panel_data.loc[(panel_data[('Attributes','Symbols','Date')] >= mask_start_time) & (panel_data[('Attributes','Symbols','Date')] <= mask_end_time)]
        copy_close= panel_data_filtered[['Close','Adj Close']]

        if option =='Volume traded':
            basic_chart = alt.Chart(df3).mark_line().encode(x='Date',y='Volume',
            color='Company',tooltip=['Date', 'Volume', 'Company'])

            st.altair_chart(basic_chart,use_container_width=False)

        if option =='Close Prices':
            basic_chart = alt.Chart(df3).mark_line().encode(x='Date',y='Close',
            color='Company',tooltip=['Date', 'Close', 'Company'])

            st.altair_chart(basic_chart,use_container_width=False)

        if option =='Adjusted Close Prices':
            basic_chart = alt.Chart(df3).mark_line().encode(x='Date',y='Adj Close',
            color='Company',tooltip=['Date', 'Adj Close', 'Company'])

            st.altair_chart(basic_chart,use_container_width=False)


        original_title = '<p style="background-color:White; font-family:Courier; text-align:center; color:Black; font-size: 30px;">DISPLAY THE RAW</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        if st.checkbox("Show"):
            st.subheader("Stock price data")
            st.write(df3)


        
    

    if visualtype == 'Annualized Volatility':

        #Inputs
        stock = st.sidebar.multiselect('Select any of the stocks for Analysis',list(tweets.Company.unique()), ['AAPL','MSFT'])
        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

       


        #Calcs with inputs
        mask_stock = df2['Company'].isin(stock)
        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')

        df2_filtered= df2[mask_stock]
        df3=df2_filtered[(df2_filtered['Date'] > mask_start_time) & (df2_filtered['Date'] < mask_end_time)]
        panel_data_filtered=panel_data.loc[(panel_data[('Attributes','Symbols','Date')] >= mask_start_time) & (panel_data[('Attributes','Symbols','Date')] <= mask_end_time)]
        copy_close= panel_data_filtered[['Close','Adj Close']]
        
        annualized_hv = np.sqrt(np.log(copy_close['Close']/copy_close['Close'].shift(1)).var()) * np.sqrt(252)
        ah=pd.DataFrame([annualized_hv]).T
        ah= ah.reset_index(level=0)
        ah= ah.set_axis(['Symbol', 'Value'], axis=1, inplace=False)
        ah_used=ah.loc[ah['Symbol'].isin(stock)]

        basic_chart2 = alt.Chart(ah_used).mark_bar().encode(x='Symbol',y='Value',)

        st.altair_chart(basic_chart2,use_container_width=False)

    if visualtype == 'Return Distribution':

        #Inputs
        stock = st.sidebar.multiselect('Select any of the stocks for Analysis',list(tweets.Company.unique()), ['AAPL','MSFT'])
        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

        #Calcs with inputs
        mask_stock = df2['Company'].isin(stock)
        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')

        df2_filtered= df2[mask_stock]
        df3=df2_filtered[(df2_filtered['Date'] > mask_start_time) & (df2_filtered['Date'] < mask_end_time)]
        panel_data_filtered=panel_data.loc[(panel_data[('Attributes','Symbols','Date')] >= mask_start_time) & (panel_data[('Attributes','Symbols','Date')] <= mask_end_time)]
        copy_close= panel_data_filtered[['Close','Adj Close']]

    

        st.subheader("Annualized Returns")
        return_series_adj = (copy_close['Adj Close'].pct_change() + 1).cumprod() -1
        annualized_return = (1 + return_series_adj.tail(1))**(1/2) - 1

        tr=annualized_return.T
        tr= tr.reset_index(level=0)
        tr= tr.set_axis(['Symbol', 'Value'], axis=1, inplace=False)
        tr_used=tr.loc[tr['Symbol'].isin(stock)]

        basic_chart3 = alt.Chart(tr_used).mark_bar().encode(x='Symbol',y='Value',)
        st.altair_chart(basic_chart3,use_container_width=False)

        st.subheader("Daily Returns")

        panel_data2=panel_data_filtered
        panel_data2.set_index(('Attributes','Symbols','Date'), inplace=True)
        daily_returns = panel_data2['Adj Close'].pct_change(1)
        daily_returns =daily_returns.transpose().reset_index().drop('level_1',axis=1)
        dr=(pd.melt(daily_returns, id_vars='level_0', var_name='Date', value_name='Value'))
        dr_used=dr.loc[dr['level_0'].isin(stock)]

        basic_chart4 = alt.Chart(dr_used).mark_line().encode(x='Date',y='Value',color='level_0')
        st.altair_chart(basic_chart4,use_container_width=False)
    
    if visualtype == 'Moving Averages':

        stock = st.sidebar.selectbox('Select any of the stocks for Analysis',list(tweets.Company.unique()))
        number = st.sidebar.slider('How many days?', 0, 25, 10)

      

        panel_data3 =panel_data
        panel_data3.set_index(('Attributes','Symbols','Date'), inplace=True)
        panel_data3 = panel_data3.droplevel(2,axis=1)
        close_ma = panel_data3['Close'][stock]

        ma_n = close_ma.rolling(window=number).mean()

        ma=pd.DataFrame([ma_n]).T
        ma= ma.reset_index(level=0)
        ma= ma.set_axis(['Date', 'Value'], axis=1, inplace=False)
        ma['label'] = 124 * ['Moving Average']

        df3=df2
        df3['label2']= 1250 * ['Close Price']
        mask_stock = df3['Company'].isin([stock])
        df_ma= df3[mask_stock]

    
        layer = alt.Chart(df_ma).mark_line().encode(
        x='Date',
        y='Close',opacity='label2')

        basic_chart5 = alt.Chart(ma).mark_line().encode(x='Date',y='Value',opacity='label')
        st.altair_chart(basic_chart5+layer,use_container_width=False)

    if visualtype == 'Correlation Plots':

        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

        #Calcs with inputs
        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')

        panel_data_filtered=panel_data.loc[(panel_data[('Attributes','Symbols','Date')] >= mask_start_time) & (panel_data[('Attributes','Symbols','Date')] <= mask_end_time)]

        panel_data4 =panel_data_filtered
        return_series_adj = (1 + panel_data4["Adj Close"].pct_change()).cumprod() -1
        return_series_adj = return_series_adj.droplevel(1,axis=1)

        cor_data = (return_series_adj.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
        cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal


        base = alt.Chart(cor_data).encode(x='variable2:O',y='variable:O')

        #Text layer with correlation labels
        #Colors are for easier readability
        text = base.mark_text().encode(text='correlation_label',color=alt.condition(
                alt.datum.correlation > 0.5, 
                alt.value('white'),
                alt.value('black')
            )
        )

        # The correlation heatmap itself
        cor_plot = base.mark_rect().encode(color='correlation:Q')

        st.altair_chart(cor_plot + text,use_container_width=True)


   

if option == 'Sentiment Analysis':


    #img4 = Image.open('./emotions.jpeg')
    #st.image(img4, width=700)
    original_title = '<p style="background-color:White; font-family:Courier; text-align:center; color:Black; font-size: 30px;">SENTIMENT ANALYSIS</p>'
    st.markdown(original_title, unsafe_allow_html=True) 
    original_title = '<p style="font-family:Courier; text-align:center; color:White; font-size: 25px;">How is the world feeling about stocks?</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    sentiment_copy=sentiment
    sentiment_copy['compound']= (sentiment_copy['positive']*1+sentiment_copy['negative']*(-1))/2


    option2 = st.sidebar.selectbox('What do you want to see?',('Various emotions','Correlation Analysis',))

    if option2 == 'Various emotions':

        stock = st.sidebar.multiselect('Select any of the stocks for Analysis',list(tweets.Company.unique()), ['AAPL','MSFT'])
        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

        option3 = st.sidebar.selectbox('Which emotion do you want to compare?',('anger', 'anticipation', 'fear', 'sadness',
       'surprise', 'trust', 'joy', 'disgust', 'positive', 'negative','compound'))

        #Calcs with inputs
        mask_stock = sentiment_copy['Company'].isin(stock)
        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')

        sentiment_filtered=sentiment_copy[mask_stock]
        sentiment2=sentiment_filtered[(sentiment_filtered['Date'] >= mask_start_time) & (sentiment_filtered['Date'] <= mask_end_time)]
        sentiment2.drop(['Unnamed: 0','Date'],inplace=True,axis=1)


        basic_chart = alt.Chart(sentiment2).mark_line().encode(x='date',y=option3,
            color='Company',)
        st.altair_chart(basic_chart,use_container_width=False)


        


    

    if option2 == 'Correlation Analysis':

        
        stock = st.sidebar.selectbox('Select any of the stocks for Analysis',list(tweets.Company.unique()),)

        start_time = st.sidebar.slider("When do you start?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")
        end_time = st.sidebar.slider("When do you end?",min_value=datetime(2021,9,1),max_value=datetime(2022,2,28),format="MM/DD/YY")

        mask_start_time=start_time.strftime("%Y-%m-%d")
        mask_end_time=end_time.strftime('%Y-%m-%d')
        emotion = st.sidebar.selectbox('Select any of the emotions',['positive','negative','compound'],)

        sentiment_copy['Sentiment'] = 1042 * ['Sentiment Score']
        sentiment_copy['Price'] = 1042 * ['Price Change']

        sentiment_filtered= sentiment_copy[sentiment_copy['Company']==stock]
        sentiment_filtered['pc']=sentiment_filtered['Adj Close'].pct_change().shift()
        sentiment2=sentiment_filtered[(sentiment_filtered['Date'] >= mask_start_time) & (sentiment_filtered['Date'] <= mask_end_time)]
        sentiment2.drop(['Unnamed: 0','Date'],inplace=True,axis=1)
        #sentiment2['date']=pd.to_datetime(sentiment2['date'])

        base = alt.Chart(sentiment2).encode(alt.X('date:O', axis=alt.Axis(title="Date")))

        l1 = base.mark_line(stroke='#57A44C').encode(
        alt.Y(emotion,axis=alt.Axis(title='Sentiment Score', titleColor='#57A44C')),)

        l2 = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(alt.Y('pc',
        axis=alt.Axis(title='Price Change', titleColor='#5276A7')))
    

        l3=alt.layer(l1, l2).resolve_scale(y = 'independent')
        st.altair_chart(l3,use_container_width=False)





    