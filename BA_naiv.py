# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np
import pandas as pd
import math
import sys
#pip install openpyxl


##global result storage:
result_matrix = pd.DataFrame()
result_per_Share = pd.DataFrame()
result_exposure = pd.DataFrame()

######  Baseline Agent naiv with no max holding Period, exposure quantity derived ######
class SimpleAgent(BaseAgent):
    def __init__(self, name: str):

        super(SimpleAgent, self).__init__(name)

        # static attributes from arguments

        self.max_spread_quoted = 0.002
        self.min_vol_quote = 10000

        # dynamic attributes
        self.spread_market = {} #dict that captures the current spread traded
        self.max_quant_ask = {} #dict, that captures the maximal tradeable quantity of the ask side
        self.max_quant_bid = {} #dict, that captures the maximal tradeable quantity of the bid side
        self.min_quant_ask = {} #dict, that captures the minimal tradeable quantity of the ask side
        self.min_quant_bid = {} #dict, that captures the minimal tradeable quantity of the bid side


        ##### Evaluation, dynamic Variables###
        self.titm = {}  # Time in the market dictionary
        self.sess_length = datetime.timedelta()  # Time the whole Backtestingsession took
        self.VWAP = {}  # stores the VWAP realized of the agent per share
        self.VWAP_Market = {}  # stores the Market VWAP realized
        self.D_V_M = {}  #accumalates dollar volume traded per share in the back testing session
        self.T_V_M = {} #accumalates volume traded per share in the back testing session
        self.VWAP_buy = {}
        self.VWAP_sell = {}
        self.VWAP_Score = {} #stores the VWAP Score of market VWAP and share VWAP
        self.trigger_storage_stop_loss = {}
        self.trigger_storage_take_prof = {}
        self.exposure_check = 0
        self.exposure_stor = [0]
        self.iterer = datetime.timedelta(seconds=30)
        self.milliseconds = datetime.timedelta(milliseconds=100)
    def on_quote(self, market_id: str, book_state: pd.Series):

        midpoint = (book_state['L1-BidPrice'] + book_state['L1-AskPrice']) / 2
        self.spread_market[market_id] = (book_state['L1-AskPrice'] - book_state['L1-BidPrice']) / midpoint
        self.min_quant_ask[market_id] = math.ceil(self.min_vol_quote / book_state['L1-BidPrice'])
        self.min_quant_bid[market_id] = math.ceil(self.min_vol_quote / book_state['L1-AskPrice'])
        self.max_quant_ask[market_id] = max(math.floor(0.95 * self.market_interface.exposure_left / book_state['L1-BidPrice']),0)
        self.max_quant_bid[market_id] = max(math.floor(0.95 * self.market_interface.exposure_left / book_state['L1-AskPrice']),0)


        trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
        trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
        # quantity per market
        quantity_buy = sum(t.quantity for t in trades_buy)
        quantity_sell = sum(t.quantity for t in trades_sell)
        quantity_unreal = quantity_buy - quantity_sell #>0 long, <0, short

        # cancel old orders
        if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
            [self.market_interface.cancel_order(order) for order in
             self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]

        if self.market_interface.exposure_left >= 50000:
            #if we dont have quotable Spread, don´t sumbitt new and cancel old:
            if self.spread_market[market_id] > self.max_spread_quoted:
                #go out of the market:
                if quantity_unreal < 0:
                    self.market_interface.submit_order(market_id, "buy", quantity=abs(quantity_unreal))
                elif quantity_unreal > 0:
                    self.market_interface.submit_order(market_id, "sell", quantity=abs(quantity_unreal))

            #if we have a quotable Spread, the agent always quotes new orders:
            else:
                book_quant = min(book_state['L1-AskSize'],book_state['L1-BidSize'])
                max_quant = min(self.max_quant_ask[market_id],self.max_quant_bid[market_id])
                min_quant = max(self.min_quant_bid[market_id], self.min_quant_ask[market_id])

                if book_quant >= min_quant and book_quant <= max_quant:
                    self.market_interface.submit_order(market_id, "buy", book_quant,
                                                       limit=book_state['L1-BidPrice'])
                    self.market_interface.submit_order(market_id, "sell", book_quant,
                                                       limit=book_state['L1-AskPrice'])
                elif book_quant < min_quant and min_quant <= max_quant:
                    self.market_interface.submit_order(market_id, "buy", min_quant,
                                                       limit=book_state['L1-BidPrice'])
                    self.market_interface.submit_order(market_id, "sell", min_quant,
                                                       limit=book_state['L1-AskPrice'])
                elif book_quant > max_quant and max_quant > min_quant:
                    self.market_interface.submit_order(market_id, "buy", max_quant,
                                                       limit=book_state['L1-BidPrice'])
                    self.market_interface.submit_order(market_id, "sell", max_quant,
                                                       limit=book_state['L1-AskPrice'])
                else:
                    pass


        #less exposure than 95% left:
        else:
            self.exposure_check = self.exposure_check + 1
            for market_id in self.market_interface.market_state_list.keys():
                trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                # quantity per market
                quantity_buy = sum(t.quantity for t in trades_buy)
                quantity_sell = sum(t.quantity for t in trades_sell)
                quantity_unreal = quantity_buy - quantity_sell  # >0 long, <0, short
                # go out of the market:
                if quantity_unreal != 0:
                    if quantity_unreal < 0:
                        self.market_interface.submit_order(market_id, "buy", quantity=abs(quantity_unreal))
                    elif quantity_unreal > 0:
                        self.market_interface.submit_order(market_id, "sell", quantity=abs(quantity_unreal))

    def on_trade(self, market_id: str, trades_state: pd.Series):

#############Evaluation################################
        ##market VWAP:
        D_V_M = sum(vol * price for vol, price in zip(trades_state.Volume, trades_state.Price))
        T_V_M = sum(trades_state.Volume)
        if market_id in self.VWAP_Market:  # to initlise VWAP market, until it has 2 value
            self.D_V_M[market_id] = self.D_V_M[market_id] + D_V_M
            self.T_V_M[market_id] = self.T_V_M[market_id] + T_V_M
        else:
            self.D_V_M[market_id] = D_V_M
            self.T_V_M[market_id] = T_V_M
        self.VWAP_Market[market_id] = self.D_V_M[market_id] / self.T_V_M[market_id]

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):

#############Evaluation################################
        ##reset iterative variables:
        if self.sess_length == datetime.timedelta(0):
            self.titm = {}  # Time in the market dictionary
            self.VWAP = {}  # stores the VWAP realized of the agent per share
            self.VWAP_Market = {}  # stores the Market VWAP realized
            self.D_V_M = {}  # accumalates dollar volume traded per share in the back testing session
            self.T_V_M = {}  # accumalates volume traded per share in the back testing session
            self.VWAP_buy = {}
            self.VWAP_sell = {}
            self.VWAP_Score = {}  # stores the VWAP Score of market VWAP and share VWAP
            self.trigger_storage_stop_loss = {}
            self.trigger_storage_take_prof = {}
            self.exposure_check = 0
            self.exposure_stor = [0]
            self.iterer = datetime.timedelta(seconds=30)
            self.milliseconds = datetime.timedelta(milliseconds=100)

        # calculate the Session length:
        delta = timestamp_next - timestamp
        self.sess_length = self.sess_length + delta
        #store every 100 miliSecond:
        if self.milliseconds < self.sess_length:
            self.milliseconds = self.milliseconds + datetime.timedelta(milliseconds=100)
             ###Calculate Time in the market per Share
            for market_id in self.market_interface.market_state_list.keys():
                if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "sell" )) and \
                    len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "buy" )) != 0:
                    if market_id not in self.titm.keys():
                        self.titm[market_id] = datetime.timedelta(0)
                    else:
                        self.titm[market_id] = self.titm[market_id] + datetime.timedelta(milliseconds=100)

        ##store exposure development

        if self.milliseconds > self.iterer:
            exposure = self.market_interface.exposure
            total_net_exposure = sum(exposure.values())
            self.exposure_stor.append(total_net_exposure)
            self.iterer = self.iterer + datetime.timedelta(seconds = 30)

        ######store values at the end of a session###
        if timestamp == timestamp_next:

            trades = self.market_interface.get_filtered_trades()
            quantity = sum(t.quantity for t in trades)
            if quantity > 0:
                self.VWAP["Total"] = sum(t.quantity * t.price for t in trades) / quantity
            else:
                self.VWAP["Total"] = 0

            for market_id in self.market_interface.market_state_list.keys():
                ###current VWAP of agent all together ###

                ###current VWAP of agent per share ###
                trades = self.market_interface.get_filtered_trades(market_id)
                quantity = sum(t.quantity for t in trades)
                if quantity > 0:
                    self.VWAP[market_id] = sum(t.quantity * t.price for t in trades) / quantity
                else:
                    self.VWAP[market_id] = 0

                ###current sell VWAP of agent per share ###
                trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                quantity_sell = sum(t.quantity for t in trades_sell)
                if quantity_sell > 0:
                    self.VWAP_sell[market_id] = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                else:
                    self.VWAP_sell[market_id] = 0

                ###current buy VWAP of agent per share ###
                trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                quantity_buy = sum(t.quantity for t in trades_buy)
                if quantity_buy > 0:
                    self.VWAP_buy[market_id] = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                else:
                    self.VWAP_buy[market_id] = 0

                ### Calculate VWAP_Score
                self.VWAP_Score[market_id] = quantity_buy * (self.VWAP_Market[market_id] - self.VWAP_buy[market_id]) + \
                                             quantity_sell * (self.VWAP_sell[market_id] - self.VWAP_Market[market_id])


            titm_df = pd.DataFrame.from_dict(self.titm, orient='index')
            proz_titm_df = titm_df / self.sess_length
            VWAP_score_df = pd.DataFrame.from_dict(self.VWAP_Score, orient='index')
            pnl_df = pd.DataFrame.from_dict(self.market_interface.pnl_realized, orient='index')
            pnl_unr_df = pd.DataFrame.from_dict(self.market_interface.pnl_unrealized, orient='index')
            trades_df = pd.DataFrame()
            volume = {}
            trading_costs = {}

            #get traded volume, costs and n of stop loss and take prof:
            for market_id in self.market_interface.market_state_list.keys():
                trades = self.market_interface.get_filtered_trades(market_id)
                quantity = sum(t.quantity * t.price for t in trades)
                volume[market_id] = quantity
                trading_costs[market_id] = quantity * self.market_interface.transaction_cost_factor

                trades_df.loc[market_id,0] = len(self.market_interface.get_filtered_trades(market_id))

                if market_id not in self.trigger_storage_take_prof:
                    self.trigger_storage_take_prof[market_id] = 0
                if market_id not in self.trigger_storage_stop_loss:
                    self.trigger_storage_stop_loss[market_id] = 0

            trigger_storage_take_prof_df = pd.DataFrame.from_dict(self.trigger_storage_take_prof, orient='index')
            trigger_storage_stop_loss_df = pd.DataFrame.from_dict(self.trigger_storage_stop_loss, orient='index')
            volume_df = pd.DataFrame.from_dict(volume, orient='index')
            trading_costs_df = pd.DataFrame.from_dict(trading_costs, orient='index')

            titm_df.columns = [str(timestamp)]
            proz_titm_df.columns = ["%-Time in the Market"]
            VWAP_score_df.columns = ["VWAP_Score"]
            pnl_df.columns = ["pnl_realized"]
            pnl_unr_df.columns = ["pnl_unrealized"]
            trades_df.columns = ["n_trades"]
            trigger_storage_take_prof_df.columns = ["n_take_prof"]
            trigger_storage_stop_loss_df.columns = ["n_stopp_loss"]
            volume_df.columns = ["dollar_volume_traded"]
            trading_costs_df.columns = ["trading_costs"]


            titm_df = titm_df.transpose()
            proz_titm_df = proz_titm_df.transpose()
            VWAP_score_df = VWAP_score_df.transpose()
            pnl_df = pnl_df.transpose()
            pnl_unr_df = pnl_unr_df.transpose()
            trades_df = trades_df.transpose()
            trigger_storage_take_prof_df = trigger_storage_take_prof_df.transpose()
            trigger_storage_stop_loss_df = trigger_storage_stop_loss_df.transpose()
            volume_df = volume_df.transpose()
            trading_costs_df = trading_costs_df.transpose()

            global result_exposure, result_per_Share, result_matrix
            result_exposure = result_exposure.append(self.exposure_stor)

            result_per_Share = result_per_Share.append([titm_df,proz_titm_df,VWAP_score_df,pnl_df,trades_df,trigger_storage_take_prof_df,
                                                        trigger_storage_stop_loss_df,volume_df,trading_costs_df,pnl_unr_df])

            result_matrix = result_matrix.append(
                {
                    "timestamp": str(timestamp),
                    "exposure": self.market_interface.exposure_total,
                    'pnl': self.market_interface.pnl_realized_total,
                    "n_trades": len(self.market_interface.get_filtered_trades()),
                    "costs": self.market_interface.transaction_cost,
                    "n_orders": len(self.market_interface.get_filtered_orders()),
                    "session_length": self.sess_length,
                    "VWAP_Total_Agent": self.VWAP["Total"],
                    "Exposure_left < 0": self.exposure_check,
                    "pnl_unrealized": self.market_interface.pnl_unrealized_total,
                }
                , ignore_index=True)




if __name__ == "__main__":


    identifier_list = [
       # ADIDAS
       "Adidas.BOOK", "Adidas.TRADES",
       # ALLIANZ
       "Allianz.BOOK", "Allianz.TRADES",
       # BASF
       "BASF.BOOK", "BASF.TRADES",
       # Bayer
       "Bayer.BOOK", "Bayer.TRADES",
       # BMW
       "BMW.BOOK", "BMW.TRADES",
       # Continental
       "Continental.BOOK", "Continental.TRADES",
       # Covestro
       "Covestro.BOOK", "Covestro.TRADES",
       # Daimler
       "Daimler.BOOK", "Daimler.TRADES",
       # Deutsche Bank
       "DeutscheBank.BOOK", "DeutscheBank.TRADES",
       # DeutscheBörse
       #"DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    agent = SimpleAgent(
        name="Bench_Agent_naiv",
    )

    backtest = Backtest(
        agent=agent, 
    )

    # Option 1: run agent against a series of generated episodes, that is, 
    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
        date_start="2021-02-01",
        date_end="2021-02-28",
        episode_interval=4, #30
        episode_shuffle=True, 
        episode_buffer=5,  #2
        episode_length=9,  ##6 length - buffer = traiding time of the agent
        num_episodes=4,  #2
        seed = 5,
    )


#print results in excel:
result_per_Share.iloc[:1] = result_per_Share.iloc[:1].astype(str)
result_matrix["session_length"] = result_matrix["session_length"].astype(str)
name_of_file = "result_" + agent.name + ".xlsx"
writer = pd.ExcelWriter(name_of_file)
result_matrix.to_excel(writer,"result_matrix")
result_per_Share.to_excel(writer,"result_per_Share")
result_exposure.to_excel(writer, "exposure_history")

writer.save()
