import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
import os



st.set_page_config(layout="wide")

st.header('Daily Tech Tweaking')


@st.cache
def avgAndSmoothPandasData(pandaFrame, avgWin=10, smthWin=4, smthStd = 5):
    rAvgSmth=pandaFrame.rolling(avgWin).mean().rolling(smthWin,win_type='gaussian').mean(std=smthStd)
    return rAvgSmth
def avgAndSmoothPandasDataMax(pandaFrame, avgWin=10, smthWin=4, smthStd = 5):
    rAvgSmth=pandaFrame.rolling(avgWin).max().rolling(smthWin,win_type='gaussian').mean(std=smthStd)
    return rAvgSmth
def avgAndSmoothPandasDataMin(pandaFrame, avgWin=10, smthWin=4, smthStd = 5):
    rAvgSmth=pandaFrame.rolling(avgWin).min().rolling(smthWin,win_type='gaussian').mean(std=smthStd)
    return rAvgSmth
def avgAndSmoothPandasDataCOV(pandaFrameA,pandFrameB, avgWin=10, smthWin=4, smthStd = 5):
	rAvgSmthCOV=pandaFrameA.rolling(avgWin).cov(pandFrameB).rolling(smthWin,win_type='gaussian').mean(std=smthStd)
	# rAvgSmthCOV=rAvgSmthCOV.fillna(method='ffill')
	return rAvgSmthCOV
def avgAndSmoothPandasDataVar(pandaFrame, avgWin=10, smthWin=4, smthStd = 5):
    rAvgSmth=pandaFrame.rolling(avgWin).var().rolling(smthWin,win_type='gaussian').mean(std=smthStd)
    return rAvgSmth


# @st.cache
def load_data(tickerNm):
	cur_workDir = os.getcwd()
	dataPath = cur_workDir + os.path.sep + 'data' + os.path.sep + '{}.csv'.format(tickerNm)
	impData=pd.read_csv(dataPath)
	impData=impData.set_index('time')
	impData=impData.fillna(method='bfill')

	newColumns = []
	for string in impData.columns:
	    newString = string.replace(", NASDAQ: ","_")
	    newString = newString.replace(", NYSE ARCA & MKT: ","_")
	    newString = newString.replace(", NYSE: ","_")
	    newString = newString.replace("open","{}_Open".format(tickerNm))
	    newString = newString.replace("high","{}_High".format(tickerNm))
	    newString = newString.replace("low","{}_Low".format(tickerNm))
	    newString = newString.replace("close","{}_Close".format(tickerNm))
	    newString = newString.replace("Volume","{}_Volume".format(tickerNm))
	    newString = newString.replace("Volume MA","{}_VolumeMA".format(tickerNm))
	    newString = newString.replace("Accumulation/Distribution","{}_AcDist".format(tickerNm))
	    newString = newString.replace("RSI","{}_RSI".format(tickerNm))
	    newColumns.append(newString)
	impData.columns=newColumns



	avgPrices=impData[[tickerNm +"_High",tickerNm +"_Low"]].mean(1)
	newColumns.append("{}_AvgPrice".format(tickerNm))

	impData=pd.concat([impData, avgPrices], axis=1)
	impData.columns=newColumns
	return impData

def compute_ADScore(acData, ad_weight):

	totalMax=avgAndSmoothPandasDataMax(acData)
	totalMin=avgAndSmoothPandasDataMin(acData)

	# totalMax=np.max(acData)
	# totalMin=np.min(acData)

	# mTSc=acData
	mTSc=1-((totalMax-acData)/(totalMax-totalMin))

	mTScTh=np.zeros(np.shape(mTSc))

	# acWeight=0.5
	mTScTh[np.where(mTSc<=0.5)[0]]=1
	mTScTh[np.where((mTSc>0.5) & (mTSc<=0.75))[0]]=2
	mTScTh[np.where((mTSc>0.75) & (mTSc<=0.85))[0]]=3
	mTScTh[np.where((mTSc>0.85) & (mTSc<=0.95))[0]]=4
	mTScTh[np.where(mTSc>0.95)[0]]=5
	adScore=mTScTh*ad_weight
	# adScore[np.where(adScore==0)[0]]=0.5
	
	return adScore

def compute_PPScore(ppData, pp_weight):
	rPDynTh=np.zeros(np.shape(ppData))

	rPDynTh[np.where(ppData<=0.01)[0]]=1
	rPDynTh[np.where((ppData>0.01) & (ppData<=0.05))[0]]=2
	rPDynTh[np.where((ppData>0.05) & (ppData<=0.10))[0]]=3
	rPDynTh[np.where((ppData>0.10) & (ppData<=0.15))[0]]=4
	rPDynTh[np.where(ppData>0.15)[0]]=5
	ppScore=rPDynTh*pp_weight
	# ppScore[np.where(ppScore==0)[0]]=0.5
	
	return ppScore

def compute_RSIScore(smthRSIData, rsi_weight):

	rPDynTh=np.zeros(np.shape(smthRSIData))
	
	

	rPDynTh[np.where(smthRSIData<=60)[0]]=1
	rPDynTh[np.where((smthRSIData>60) & (smthRSIData<=80))[0]]=2
	rPDynTh[np.where((smthRSIData>80) & (smthRSIData<=90))[0]]=3
	rPDynTh[np.where((smthRSIData>90) & (smthRSIData<=95))[0]]=4
	rPDynTh[np.where(smthRSIData>95)[0]]=5
	rsiScore=rPDynTh*rsi_weight
	# rsiScore[np.where(rsiScore==0)[0]]=0.5
	
	return rsiScore
def compute_VOLScore(smthVolData, vol_weight):

	rPDynTh=np.zeros(np.shape(smthVolData))


	rPDynTh[np.where(smthVolData<=0.75)[0]]=1
	rPDynTh[np.where((smthVolData>0.75) & (smthVolData<=1.0))[0]]=2
	rPDynTh[np.where((smthVolData>1.0) & (smthVolData<=2.0))[0]]=3
	rPDynTh[np.where((smthVolData>2.0) & (smthVolData<=2.5))[0]]=4
	rPDynTh[np.where(smthVolData>2.5)[0]]=5
	volScore=rPDynTh*vol_weight
	# volScore[np.where(volScore==0)[0]]=0.5
	
	return volScore

def compute_betaScore(smthBetaData, beta_weight):

	rPDynTh=np.zeros(np.shape(smthBetaData))
	
	

	rPDynTh[np.where(smthBetaData<=0.75)[0]]=1
	rPDynTh[np.where((smthBetaData>0.75) & (smthBetaData<=1.0))[0]]=2
	rPDynTh[np.where((smthBetaData>1.0) & (smthBetaData<=2.0))[0]]=3
	rPDynTh[np.where((smthBetaData>2.0) & (smthBetaData<=2.5))[0]]=4
	rPDynTh[np.where(smthBetaData>2.5)[0]]=5
	betaScore=rPDynTh*beta_weight
	# betaScore[np.where(betaScore==0)[0]]=0.5
	
	return betaScore

# data_load_state = st.text('Loading data...')

firstTicker = st.sidebar.selectbox('Select Available Ticker',(['AAPL','AFRM','BEAM','CRM','ECPG','LABU','LRCX','NTLA','PLAB','RH','TNA','VRTX','WCLD']), index=0)



data   = load_data(firstTicker)
# st.subheader('Raw data')
# st.write(data)


mvAvgBin = st.sidebar.slider('# moving average bins (default = 10)', 0, 50, 10,key=1)
gausBin = st.sidebar.slider('width of smoothing (default = 4)', 0, 25, 4,key=2)
gausStd = st.sidebar.slider('std of smooth (default = 5)', 0, 50, 5,key=3)

ppW=st.sidebar.number_input(label='price perf weight',value=0.3, format='%f')
adW=st.sidebar.number_input(label='acc/dist weight',value=0.5, format='%f')
rsiW=st.sidebar.number_input(label='rsi weight',value=0.1, format='%f')
vlmW = st.sidebar.number_input(label='volume weight',value=0.05, format='%f')
betaW=st.sidebar.number_input(label='beta weight',value=0.05, format='%f')



col1, col2, col3, col4, col5 = st.columns(5)
with col1:
	spyData = load_data('SPY')
	# st.header('Avg Price')

	smthData = avgAndSmoothPandasData(data['{}_AvgPrice'.format(firstTicker)],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	opPrice=data['{}_AvgPrice'.format(firstTicker)][0:mvAvgBin].mean()
	rollingPercent=(smthData-opPrice)/opPrice
	
	spyDataAvg=pd.concat([data['{}_AvgPrice'.format(firstTicker)],spyData['SPY_AvgPrice']], axis=1)
	spyDataAvg=spyDataAvg.dropna(axis=0)
	# # st.subheader('Raw data')
	# st.write(betaD)

	betaD=avgAndSmoothPandasDataCOV(spyDataAvg['SPY_AvgPrice'],spyDataAvg['{}_AvgPrice'.format(firstTicker)],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	betaD=betaD/avgAndSmoothPandasDataVar(spyDataAvg['SPY_AvgPrice'],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	smthRSI = avgAndSmoothPandasData(data['{}_RSI'.format(firstTicker)],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	smthVolume = avgAndSmoothPandasData(data['{}_Volume'.format(firstTicker)],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	opVol=data['{}_Volume'.format(firstTicker)][0:mvAvgBin].mean()
	rollingVolPercent=(smthVolume-opVol)/opVol

	# # st.subheader('Raw data')
	# st.write(betaD)
	x = [np.arange(0,len(data['{}_AvgPrice'.format(firstTicker)].values)),np.arange(0,len(data['{}_AvgPrice'.format(firstTicker)].values))]
	#y = [data['{}_AvgPrice'.format(firstTicker)],smthData]
	y = [data['{}_AvgPrice'.format(firstTicker)],smthData]

	
	
	# y=[rollingPercent,rollingPercent]

	p1 = figure(
	     title='avg price',
	     x_axis_label='minutes',
	     y_axis_label='avg price',width=400, height=300)

	p1.multi_line(x, y, line_width=2, line_color=['black','blue'])
	st.bokeh_chart(p1, use_container_width=True)

with col2:

	smthData_acDist = avgAndSmoothPandasData(data['{}_AcDist'.format(firstTicker)],avgWin=mvAvgBin,smthWin=gausBin, smthStd = gausStd)
	x = [np.arange(0,len(data['{}_AcDist'.format(firstTicker)].values)),np.arange(0,len(data['{}_AcDist'.format(firstTicker)].values))]
	y = [data['{}_AcDist'.format(firstTicker)].values,smthData_acDist]

	p2 = figure(
	     title='acc & dist',
	     x_axis_label='minutes',
	     y_axis_label='acc/dist',width=400, height=300)

	p2.multi_line(x, y, line_width=2, line_color=['black','blue'])
	st.bokeh_chart(p2, use_container_width=True)

with col3:

	x = [np.arange(0,len(data['{}_RSI'.format(firstTicker)].values)),np.arange(0,len(data['{}_RSI'.format(firstTicker)].values))]
	y = [data['{}_RSI'.format(firstTicker)].values,smthRSI]

	p3 = figure(
	     title='RSI',
	     x_axis_label='minutes',
	     y_axis_label='RSI',width=400, height=300)

	p3.multi_line(x, y, line_width=2, line_color=['black','blue'])
	st.bokeh_chart(p3, use_container_width=True)

with col4:

	x = [np.arange(0,len(data['{}_Volume'.format(firstTicker)].values)),np.arange(0,len(data['{}_Volume'.format(firstTicker)].values))]
	y = [rollingVolPercent]

	p4 = figure(
	     title='Rel. Volume Change',
	     x_axis_label='minutes',
	     y_axis_label='frac of shares',width=400, height=300)

	p4.multi_line(x, y, line_width=2, line_color=['black','blue'])
	st.bokeh_chart(p4, use_container_width=True)

with col5:

	x = [np.arange(0,len(data['{}_AvgPrice'.format(firstTicker)].values))]
	y = [betaD]

	p5 = figure(
	     title='Volatility',
	     x_axis_label='minutes',
	     y_axis_label='beta',width=400, height=300)

	p5.multi_line(x, y, line_width=2, line_color=['blue'])
	st.bokeh_chart(p5, use_container_width=True)

with col1:
	prelimPPScre=compute_PPScore(rollingPercent,ppW)
	x = np.arange(0,len(prelimPPScre))
	y = prelimPPScre
	#y = prelimPPScre ppW

	p6 = figure(
	     title='pp score: max = {}'.format(ppW*5),
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300, y_range=(0, ppW*5+0.1))

	p6.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p6, use_container_width=True)

with col2:
	prelimADScre=compute_ADScore(smthData_acDist,adW)
	x = np.arange(0,len(prelimADScre))
	y = prelimADScre
	#y = prelimPPScre

	p7 = figure(
	     title='ad score: max = {}'.format(adW*5),
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300, y_range=(0, adW*5+0.1))

	p7.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p7, use_container_width=True)

with col3:
	prelimRSIScre=compute_RSIScore(smthRSI,rsiW)
	x = np.arange(0,len(prelimRSIScre))
	y = prelimRSIScre
	#y = prelimPPScre

	p8 = figure(
	     title='rsi score: max = {}'.format(rsiW*5),
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300, y_range=(0, rsiW*5+0.1))

	p8.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p8, use_container_width=True)

with col4:
	prelimVlmScre=compute_VOLScore(rollingVolPercent,vlmW)
	x = np.arange(0,len(prelimVlmScre))
	y = prelimVlmScre
	#y = prelimPPScre

	p9 = figure(
	     title='volume score: max = {}'.format(vlmW*5),
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300, y_range=(0, vlmW*5+0.1))

	p9.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p9, use_container_width=True)

with col5:
	prelimBetaScre=compute_betaScore(betaD,betaW)
	x = np.arange(0,len(prelimBetaScre))
	y = prelimBetaScre
	#y = prelimPPScre

	p9 = figure(
	     title='beta score: max = {}'.format(betaW*5),
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300, y_range=(0, betaW*5+0.1))

	p9.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p9, use_container_width=True)

with col1:
	
	x = np.arange(0,len(prelimBetaScre))
	y = prelimBetaScre+prelimVlmScre+prelimRSIScre+prelimADScre+prelimPPScre
	#y = prelimPPScre

	p10 = figure(
	     title='total score',
	     x_axis_label='minutes',
	     y_axis_label='score',width=400, height=300,y_range=(0, 5+0.1))

	p10.line(x, y, line_width=2, line_color='black')
	st.bokeh_chart(p10, use_container_width=True)




