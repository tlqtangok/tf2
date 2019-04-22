### success ok ### 

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 
from IPython.display import display

import tushare as ts

import datetime 

def append_text_to_file(id_text,id_filename): 
    out = open(id_filename, 'a')
    out.write(id_text)
    out.close()
    
def chomp(id_str):
    return id_str.strip()

def from_code_to_name(df_astock_9, s_code):
    return df_astock_9[df_astock_9["code"]== s_code]["name"].ravel()[0]
    
def get_each_stock_hist_may_save(stock_code,  range_days):
    fn = str(stock_code) + ".csv"
    et = "E:/jd/t"
    fn_full = et + "/" + fn
    
    
    date_now = datetime.datetime.now()
    date_from = date_now - datetime.timedelta(range_days)
    
    date_now_f = f"{date_now:%Y-%m-%d}"
    date_start_f = f"{date_from:%Y-%m-%d}"
    
#     print(date_start_f)
    e_hist = None
#     print(fn_full)
    
    if os.path.exists(fn_full):
#         print (fn_full + " exists")
        e_hist = pd.read_csv(fn_full, dtype={"code":str})
        e_hist.set_index("date", inplace=True)
        
        
    else:
        e_hist = ts.get_hist_data(stock_code,  start=date_start_f, end=date_now_f) 
        
        try:
        
#             print (e_hist.head())
            e_hist.to_csv(fn_full)
        except:
            e_hist = pd.DataFrame()
#     display(e_hist.head())
    return e_hist

    
def run_algo_on_each_hist(e_hist, stock_code, stock_name):
    
#     id_pd = pd.read_csv(fn_full, encoding="gb2312", header="infer", sep=",")
    id_pd = e_hist
    et = "E:/jd/t"
    sc_full = et + "/" + "sc.PL"
    
    
#     print(id_pd.index[0])
    current_close_price = id_pd.loc[id_pd.index[0]]["close"]
    
#     print(current_close_price)
    
    
#     id_pd_ = (id_pd.head(90)[::-1])
#     id_pd_ = id_pd.head(range_days)
    id_pd_ = id_pd
    
    to_date = id_pd_.loc[id_pd.index[0]].name
    
    range_days_ = len(id_pd_)
    
#     print(range_days_)
    
    from_date = id_pd_.loc[id_pd.index[range_days_-1]].name
    
    
#     display(id_pd_.head(3))

    
    
#     display(id_pd_)
    low_3_ele = id_pd_["low"].nsmallest(3)
#     print(low_3_ele)
    
    
    
    low_index = low_3_ele.index[1]
    
#     print(low_index)
#     print (id_pd_.loc[:low_index])
    
    low = low_3_ele.ravel()[1]
#     print ("- low ", low)
    
    
#     print(id_pd_.index[0:low_index])
    
    
    id_pd_cut_with_low = id_pd_.loc[:low_index]
    
#     display(id_pd_cut_with_low.tail(3))
    
    high_3_ele = id_pd_cut_with_low["high"].nlargest(3)
#     display(high)
    
#     print(high.index[1])
    
    
    high = high_3_ele.ravel()[1]
    
    
    id_cmd_run_sc = " perl " + sc_full + " " + str(low) + " " + str(high)
    
#     print (id_cmd_run_sc)
    
    range_buy_sell = os.popen(id_cmd_run_sc).read()
    range_buy_sell = chomp(range_buy_sell)
    
    

    
    arr_bs = range_buy_sell.split(' ~ ')
    buy_low = float(arr_bs[0])
    buy_high = float(arr_bs[1])
    
    id_str_ret = ("- bs: [ {:s} => {:s} ]   ---   [stock: {:s},{:s}, current: {:.2f}], date: [{:10s} => {:10s}], price: [{:4.2f}, {:4.2f}]\t\t".format(arr_bs[0], arr_bs[1],stock_code, stock_name, current_close_price, from_date, to_date, low, high))    # python3 
    
#     print(current_close_price)
    if buy_high >= current_close_price:
        id_str_ret += " - stock "+ str(stock_code) + " can buy now ! current price is " + str(current_close_price)
        if buy_low >= current_close_price:
            id_str_ret += " - super recommend!!! stock " + str(stock_code) + " can buy buy buy now !current price is " +  str(current_close_price)
    
    else:
        if buy_high * 1.03 >= current_close_price:
            id_str_ret += "- stock "+ str(stock_code) + " is ask for a monitor ! current price is " + str(current_close_price)+ ", only 3% !"

    return id_str_ret + "\n"

def run_loop_get_stock_days(stock_code_list, range_days=90):
    
#     fn = str(stock_code) + ".csv"
    et = "E:/jd/t"
#     fn_full = et + "/" + fn
    sc_full = et + "/" + "sc.PL"
    fn_log = "stock_s.log"
    
    fn_log_full = et + "/" + fn_log
    
    fn_astock_all = "astock_all.csv"
    
    date_now = datetime.datetime.now()
    date_from = date_now - datetime.timedelta(days=range_days)
    
    date_now = f"{date_now:%Y_%m_%d}"
    date_from = f"{date_from:%Y_%m_%d}"
    df_astock = None
    
    if os.path.exists(fn_astock_all):
        df_astock = pd.read_csv(fn_astock_all, dtype={"code":str})
        
    else:
        df_astock = ts.get_stock_basics()
        
    
        
    
#     df_astock_9 = df_astock.head(9)
    
#     t = df_astock_9[df_astock_9["code"]== "300325"]["name"].ravel()[0]
  
    fp = open(fn_log_full, "wb")
    fp.close()
    
    if len(stock_code_list) == 0:
        stock_code_list = df_astock["code"].ravel()
    
#     print ("- len of all a stocks : " + str(len(stock_code_list)))
    
    cnt = 0
    for e_code in stock_code_list:
        print ("- cnt : " + str(cnt))
        
        stock_name = from_code_to_name(df_astock, e_code)
        
#         print(e_code, from_code_to_name(df_astock, e_code))
        e_hist = get_each_stock_hist_may_save(e_code,range_days)
#         print(e_hist.shape[0])
        
      
        if e_hist.shape[0] < range_days / 2.0:
            continue
            
#         display(e_hist)
        e_ret_str = "NULL"
        
        try:
            e_ret_str = run_algo_on_each_hist(e_hist, e_code, stock_name)
            print(e_ret_str)
            append_text_to_file(e_ret_str, fn_log_full)
        except:
            print("- error on stock " + e_code)
        cnt = cnt + 1
        

    
    
    
    
if __name__ == "__main__":
    !perl -e "print time; "
    print()

    stock_code_list = []
    
#     stock_code_list = ["300676", "000404", "000802"]
#     stock_code_list = [ '600145', '600126', '600115', '600103', '600095', '600077', '600008', '300726']
    
    
    run_loop_get_stock_days(stock_code_list, 188)
    print()
    
    !perl -e "print time;"


    




# ts.get_stock_basics()
# ts.get_hist_data('000404',  start="2018-01-01") #深圳综合指数



