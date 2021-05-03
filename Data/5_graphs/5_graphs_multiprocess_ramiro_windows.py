# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:25:23 2021

@author: m.paula.basigalup
"""

import glob
import os
import random
from multiprocessing import Pool, freeze_support
import functools
import tqdm

import mplfinance as mpf
import pandas as pd
from pyti.bollinger_bands import lower_bollinger_band as lower_b
from pyti.bollinger_bands import middle_bollinger_band as mid_b
from pyti.bollinger_bands import upper_bollinger_band as upper_b

random.seed(123)

wd = 'C:/Users/BASIGALUP/Documents/MiMA/Tesis/Candlestick Analysis/Data'
owd = wd + '/4_classification'

# Define functions
def obtain_data(iwd, file, start=None, end=None):
    # Dates are optional, if not entered, all data will be read
    # Enter the start and end dates using the method date(yyyy,m,dd)
    df = pd.read_csv(iwd + '/' + file)
    # df = pd.read_csv(owd + '/' + ticker + '.csv')
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    if start:
        df = df[df['Date'] >= start]
    if end:
        df = df[df['Date'] <= end]
    # df.index=df.Date
    return df


def plot_candles_mpf(
    df,
    k_period,
    trend,
    mav,
    bollinger,
    volume,
    show_nontrading,
    axisoff,
    fwd,
    mav_values=None,
    bollinger_period=None,
):

    ticker = df.ticker.unique()[0]
    graph_date = df.index[-1].date()

    mc = mpf.make_marketcolors(up='limegreen', down='red', volume='gray', ohlc='black')
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        mavcolors=['coral', 'magenta', 'deepskyblue'],
    )
    # style = 'yahoo'
    save = dict(
        fname=fwd
        + '/'
        + trend
        + '/'
        + ticker
        + '_'
        + str(k_period)
        + '_'
        + str(graph_date)
        + '.jpg'
    )

    if mav == False:
        mav_values = ()

    if bollinger == True:
        # Bollinger Bands Calculation
        data = df['Close'].values.tolist()
        bb_up = upper_b(data, bollinger_period)
        bb_mid = mid_b(data, bollinger_period)
        bb_low = lower_b(data, bollinger_period)
        df['bb_up'] = bb_up
        df['bb_mid'] = bb_mid
        df['bb_low'] = bb_low
        # Add Bollinger bands
        add_up_low = mpf.make_addplot(
            df[['bb_up', 'bb_mid', 'bb_low']], color='#3838ea', alpha=0.50, width=1
        )
        add_mid = mpf.make_addplot(df[['bb_mid']], color='orange', alpha=0.50, width=1)
        # Display
        # mc = mpf.make_marketcolors(up='g',down='r')
        # s  = mpf.make_mpf_style(marketcolors=mc)
        fill_between = dict(
            y1=df['bb_low'].values, y2=df['bb_up'].values, color='#f2ad73', alpha=0.20
        )

        return mpf.plot(
            df,
            type='candle',
            mav=mav_values,
            volume=volume,
            style=s,
            addplot=[add_up_low, add_mid],
            fill_between=fill_between,
            show_nontrading=show_nontrading,
            axisoff=axisoff,
            savefig=save,
        )

    else:
        return mpf.plot(
            df,
            type='candle',
            mav=mav_values,
            volume=volume,
            style=s,
            show_nontrading=show_nontrading,
            axisoff=axisoff,
            savefig=save,
        )


def generate_graphs(iwd, file, n_obs, k_period, fwd):

    mav = False
    mav_values = (3, 6, 9)
    bollinger = False
    bollinger_period = 9
    volume = True
    show_nontrading = False
    axisoff = True

    df_original = obtain_data(iwd, file)
    df_original.index = pd.to_datetime(df_original.Date)
    del df_original['Date']
    df_filtered = df_original.copy()
    df_filtered.reset_index(inplace=True)
    df_filtered = df_filtered.dropna(subset=['Trend'])
    year_list = pd.to_datetime(df_filtered['Date']).dt.year.unique()
    date_list = df_filtered.loc[k_period - 1 :, 'Date']
    obs_list = []

    for year in year_list:
        year_date_list = [date for date in date_list if date.year == year]
        year_obs_list = random.sample(year_date_list, n_obs)
        obs_list = obs_list + year_obs_list

    all_df_obs = []
    for obs in obs_list:
        # print(obs)
        end = obs
        end_i = df_filtered[df_filtered['Date'] == obs].index.item()
        start_i = end_i - (k_period - 1)
        start = df_filtered['Date'][start_i]
        trend = df_filtered['Trend'][end_i]
        df_obs = df_original.loc[start:end]
        all_df_obs.append(df_obs)
    
    plot_candles_mpf_partial = functools.partial(
        plot_candles_mpf,
        k_period = k_period,
        trend = trend,
        mav = mav,
        bollinger = bollinger,
        volume = volume,
        show_nontrading = show_nontrading,
        axisoff = axisoff,
        fwd = fwd,
        mav_values = mav_values,
        bollinger_period = bollinger_period,
    )

    with Pool(processes=8) as p:
        _= p.map(plot_candles_mpf_partial, all_df_obs)


def main():
    n_obs = 36
    k_period = 5

    # Generate graphs
    subfolders = [f.name for f in os.scandir(owd) if f.is_dir()]

    for folder in subfolders:
        print(folder)
        iwd = wd + '/4_classification' + "/" + folder
        fwd = wd + '/5_graphs' + "/" + folder
        os.chdir(iwd)
        all_files = [i for i in glob.glob('*.{}'.format('csv'))]

        for f in tqdm.tqdm(all_files):
            generate_graphs(iwd, f, n_obs, k_period, fwd)


if __name__ == "__main__":
    freeze_support()
    main()