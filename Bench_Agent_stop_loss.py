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

episode_list=[
            ("2021-01-05T08:00:00", "2021-01-05T08:30:00", "2021-01-05T08:35:00"),
            #("2021-01-06T08:00:00", "2021-01-06T015:30:00", "2021-01-06T16:00:00"),
            #("2021-02-18T12:00:00", "2021-02-18T12:15:00", "2021-02-18T12:30:00"),
            #("2021-01-05T12:00:00", "2021-01-05T12:30:00", "2021-01-05T12:32:00"),
        ]

##global result storage:
result_matrix = pd.DataFrame()
result_per_Share = pd.DataFrame()
trades= {}
orders= {}

######  Baseline Agent  ######
"""
-The Baseline/Benchmark Agent follows simple Rule Based MM-Strategy: Always trys to place best Bid and best Ask Order from the last Order book Update
buy taking the min of (L1quant_Bid, L1_quantAsk) as th quantity traded (to ensure, zero net Position is build) and L1 Price as a Order imput
-to account for a MM-strategie, that only trys to earn the spread, a stop loss,take profit is set and holding an exposure !=0 per Share
should be smaller than 1 seconds

Rule:
- always place a "best"-quote (ask(1) and bid(1) from last Book update at the same time)
- if both orders are executed, new quote will be set
- if only one side is executed, the other will be canceled and a new quote is set
- if stop loss/take profit is triggered, come back to a net exposure = 0 (possible improvement, only cancel on side take profit and loss --> momentum)
- minimum quantity submitted total per share is always the rounded up value of 10.000€ / Bid(ask)
(10.000/share Price) --> only minimum volume is traded, dynamic, because share Price is dynamic
- quantity submitted is ether the quantity of best level on the other side, if quantity > minimum quantity, or minimum quantity it self
- closes all Position at 16.25, 5 min before ending
- Always check, if there is enough exposure left and stop loss is triggered


 """



class SimpleAgent(BaseAgent):
    def __init__(self, name: str, take_prof: float, stop_loss: float, max_holding_period):

        super(SimpleAgent, self).__init__(name)

        # static attributes from arguments
        self.take_prof = take_prof
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period

        # further static attributes
        self.end_time = datetime.time(16, 25)
        self.start_time = datetime.time(8, 5)
        self.max_vol_difference = 0.5
        self.max_spread_quoted = 0.002
        self.min_vol_quote = 10000

        # dynamic attributes
        self.spread_quoted = {}  # dict that captures the quoted spread of the agent
        self.spread_market = {} # dict that captures the current spread traded
        self.trading_phase = False  # indicates whether algo is ready to trade
        self.best_ask = {}  # dict that captures ask_quantity submitted of each market
        self.best_bid = {}  # dict that captures bid_quantity submitted of each market
        self.midpoint = {}
        self.max_quant_ask = {} #dict, that captures the maximal tradeable quantity of the ask side
        self.max_quant_bid = {} #dict, that captures the maximal tradeable quantity of the bid side
        #self.min_quant_ask = {} #dict, that captures the maximal tradeable quantity of the ask side
        #self.min_quant_bid = {} #dict, that captures the maximal tradeable quantity of the bid side
        #self.possible_exposure = 0  #irreative varible, tracking if all orders together exceed the endowment
        self.holding_time_buy = {} #dict, checking how long the Agent is in the market
        self.holding_time_sell = {}

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
        self.trigger_storage_max_hold = {}



    def on_quote(self, market_id: str, book_state: pd.Series):

        if self.trading_phase:

            ##calculate best bid/ask in the order book per share:
            self.best_ask[market_id] = book_state['L1-AskPrice']
            self.best_bid[market_id] = book_state['L1-BidPrice']
            self.midpoint[market_id] = (book_state['L1-BidPrice'] + book_state['L1-AskPrice']) / 2
            self.spread_market[market_id] = (book_state['L1-AskPrice'] - book_state['L1-BidPrice']) / self.midpoint[market_id]
            #self.min_quant_ask[market_id] = math.ceil(self.min_vol_quote / book_state['L1-BidPrice'])
            #self.min_quant_bid[market_id] = math.ceil(self.min_vol_quote / book_state['L1-AskPrice'])
            self.max_quant_ask[market_id] = max(math.floor(self.market_interface.exposure_left / book_state['L1-BidPrice']),0)
            self.max_quant_bid[market_id] = max(math.floor(self.market_interface.exposure_left / book_state['L1-AskPrice']),0)

            trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
            trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
            # quantity per market
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell #>0 long, <0, short
            timestamp = book_state.TIMESTAMP_UTC

            if self.market_interface.exposure_left >= 0:
                #if we dont have quotable Spread, don´t sumbitt new and cancel old:
                if self.spread_market[market_id] > self.max_spread_quoted:
                    #cancel all orders
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    #go out of the market:
                    if quantity_unreal < 0:
                        self.market_interface.submit_order(market_id, "buy", quantity=abs(quantity_unreal))
                    elif quantity_unreal > 0:
                        self.market_interface.submit_order(market_id, "sell", quantity=abs(quantity_unreal))

                #if we have a quotable Spread, the agent always quotes new orders:
                else:
                    #delete old orders:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]


                    #submit L1 order for buy and sell, if we have no volume difference:
                    min_quant = min(book_state['L1-AskSize'],book_state['L1-BidSize']) #use min quant, that no position is generated, only spread is earned
                    #buy:
                    if min_quant <= self.max_quant_ask[market_id]:
                        self.market_interface.submit_order(market_id, "buy", min_quant,
                                                           limit=book_state['L1-BidPrice'])
                    elif min_quant > self.max_quant_ask[market_id]:
                        self.market_interface.submit_order(market_id, "buy", self.max_quant_ask[market_id],
                                                           limit=book_state['L1-BidPrice'])

                    #sell:
                    if min_quant <= self.max_quant_bid[market_id]:
                        self.market_interface.submit_order(market_id, "sell", min_quant,
                                                           limit=book_state['L1-AskPrice'])
                    elif min_quant > self.max_quant_bid[market_id]:
                        self.market_interface.submit_order(market_id, "sell", self.max_quant_bid[market_id],
                                                           limit=book_state['L1-AskPrice'])
                    else:
                        pass

            #no exposure left (dürfte nie eintreten):
            else:
                sys.exit("exposure dürfte nie unter 0 fallen")

#### check for take profit or stop loss:

            # for Long positions:
            if quantity_unreal > 0:
                exec_price = self.market_interface.get_filtered_trades(market_id, side="buy")[-1].price
                trade_profit = book_state['L1-BidPrice']/exec_price - 1
                # long --> want to sell (ask), gets matched with bid --> Bid is counter part

                if trade_profit * -1 > self.stop_loss:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "sell", quantity= abs(quantity_unreal))
                    if market_id in self.trigger_storage_stop_loss:
                        self.trigger_storage_stop_loss[market_id] = self.trigger_storage_stop_loss[market_id] + 1
                    else:
                        self.trigger_storage_stop_loss[market_id] = 1

                if trade_profit > self.take_prof:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "sell", quantity= abs(quantity_unreal))
                    if market_id in self.trigger_storage_take_prof:
                        self.trigger_storage_take_prof[market_id] = self.trigger_storage_take_prof[market_id] + 1
                    else:
                        self.trigger_storage_take_prof[market_id] = 1

            # for short positions:
            if quantity_unreal < 0:
                exec_price = self.market_interface.get_filtered_trades(market_id, side="sell")[-1].price
                trade_profit = -1 * (book_state['L1-AskPrice']/exec_price - 1)
                if trade_profit * -1 > self.stop_loss:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "buy", quantity= abs(quantity_unreal))
                    if market_id in self.trigger_storage_stop_loss:
                        self.trigger_storage_stop_loss[market_id] = self.trigger_storage_stop_loss[market_id] + 1
                    else:
                        self.trigger_storage_stop_loss[market_id] = 1

                if trade_profit > self.take_prof:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "sell", quantity= abs(quantity_unreal))
                    if market_id in self.trigger_storage_take_prof:
                        self.trigger_storage_take_prof[market_id] = self.trigger_storage_take_prof[market_id] + 1
                    else:
                        self.trigger_storage_take_prof[market_id] = 1

    def on_trade(self, market_id: str, trades_state: pd.Series):


#######################################################
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

        ###current VWAP of agent all together ###
        trades = self.market_interface.get_filtered_trades()
        quantity = sum(t.quantity for t in trades)
        if quantity > 0:
            self.VWAP["Total"] = sum(t.quantity * t.price for t in trades) / quantity
        else:
            self.VWAP["Total"] = 0

        ###current VWAP of agent per share ###
        trades = self.market_interface.get_filtered_trades(market_id)
        quantity = sum(t.quantity for t in trades)
        if quantity > 0:
            self.VWAP[market_id] = sum(t.quantity * t.price for t in trades) / quantity
        else:
            self.VWAP[market_id] = 0

        ###current sell VWAP of agent per share ###
        trades_sell = self.market_interface.get_filtered_trades(market_id,side = "sell")
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
        self.VWAP_Score[market_id] = quantity_buy * (self.VWAP_Market[market_id] - self.VWAP_buy[market_id]) +\
                                     quantity_sell * (self.VWAP_sell[market_id] - self.VWAP_Market[market_id])


    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):

        #calculate the Session length:
        delta_sess = timestamp_next - timestamp
        self.sess_length = self.sess_length + delta_sess

        ### Check, if we are in Trading phase:
        trading_time = timestamp.time() > self.start_time and \
                       timestamp.time() < self.end_time

        if trading_time and not self.trading_phase:
            print('Algo is now able to trade...')
            self.trading_phase = True

        # Close, if trading phase = false:
        elif not trading_time and self.trading_phase:

            for market_id in self.market_interface.market_state_list.keys():

                # cancel active orders for this market
                if len(self.market_interface.get_filtered_orders(market_id,status="ACTIVE")) > 0:
                    [self.market_interface.cancel_order(order) for order in
                    self.market_interface.get_filtered_orders(market_id,status="ACTIVE")]

                # close positions for this market
                trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                # quantity per market
                quantity_buy = sum(t.quantity for t in trades_buy)
                quantity_sell = sum(t.quantity for t in trades_sell)
                quantity_unreal = quantity_buy - quantity_sell

                if quantity_unreal > 0:
                    self.market_interface.submit_order(
                        market_id, "sell", abs(quantity_unreal))
                if quantity_unreal < 0:
                    self.market_interface.submit_order(
                        market_id, "buy", abs(quantity_unreal))
            self.trading_phase = False

        ###check holding time, how long do i have a long or short position?#####
        delta = timestamp_next - timestamp
        for market_id in self.market_interface.market_state_list.keys():
            trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
            trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
            quantity_buy = sum(t.quantity for t in trades_buy)
            quantity_sell = sum(t.quantity for t in trades_sell)
            quantity_unreal = quantity_buy - quantity_sell  # >0 long, <0, short

            if quantity_unreal > 0:
                if market_id not in self.holding_time_buy.keys():
                    self.holding_time_buy[market_id] = datetime.timedelta(0)
                else:
                    self.holding_time_buy[market_id] = self.holding_time_buy[market_id] + delta

            elif quantity_unreal < 0:
                if market_id not in self.holding_time_sell.keys():
                    self.holding_time_sell[market_id] = datetime.timedelta(0)
                else:
                    self.holding_time_sell[market_id] = self.holding_time_sell[market_id] + delta
            elif quantity_unreal == 0:
                self.holding_time_sell[market_id] = datetime.timedelta(0)
                self.holding_time_buy[market_id] = datetime.timedelta(0)

            ###close all Positions:
            if self.holding_time_sell[market_id] > self.max_holding_period or \
                    self.holding_time_buy[market_id] > self.max_holding_period:
                if quantity_unreal < 0:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "buy", quantity= abs(quantity_unreal))
                elif quantity_unreal > 0:
                    if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
                        [self.market_interface.cancel_order(order) for order in
                        self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]
                    self.market_interface.submit_order(market_id, "sell", quantity= abs(quantity_unreal))
            else:
                pass



#######################################################
#############Evaluation################################

        ###Calculate Time in the market per Share
        for market_id in self.market_interface.market_state_list.keys():
            if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "sell" )) and \
                len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "buy" )) != 0:
                if market_id not in self.titm.keys():
                    self.titm[market_id] = datetime.timedelta(0)
                else:
                    delta = timestamp_next - timestamp
                    self.titm[market_id] = self.titm[market_id] + delta


        ######store values at the end of a session###
        if timestamp == timestamp_next:

            global trades
            t = {"timestamp": self.market_interface.get_filtered_trades()}
            trades.update(t)
            global orders
            o = {"timestamp": self.market_interface.get_filtered_orders()}
            orders.update(o)

            titm_df = pd.DataFrame.from_dict(self.titm, orient='index')
            proz_titm_df = titm_df / self.sess_length
            VWAP_score_df = pd.DataFrame.from_dict(self.VWAP_Score, orient='index')
            pnl_df = pd.DataFrame.from_dict(self.market_interface.pnl_realized, orient='index')
            trades_df = pd.DataFrame()

            for market_id in self.market_interface.market_state_list.keys():
                trades_df.loc[market_id,0] = len(self.market_interface.get_filtered_trades(market_id))

                if market_id not in self.trigger_storage_take_prof:
                    self.trigger_storage_take_prof[market_id] = 0
                if market_id not in self.trigger_storage_stop_loss:
                    self.trigger_storage_stop_loss[market_id] = 0
                if market_id not in self.trigger_storage_max_hold:
                    self.trigger_storage_max_hold[market_id] = 0

            trigger_storage_take_prof_df = pd.DataFrame.from_dict(self.trigger_storage_take_prof, orient='index')
            trigger_storage_stop_loss_df = pd.DataFrame.from_dict(self.trigger_storage_stop_loss, orient='index')
            trigger_storage_max_hold_df = pd.DataFrame.from_dict(self.trigger_storage_max_hold, orient='index')


            titm_df.columns = [str(timestamp)]
            proz_titm_df.columns = ["%-Time in the Market"]
            VWAP_score_df.columns = ["VWAP_Score"]
            pnl_df.columns = ["pnl_realaized"]
            trades_df.columns = ["n_trades"]
            trigger_storage_take_prof_df.columns = ["n_take_prof"]
            trigger_storage_stop_loss_df.columns = ["n_stopp_loss"]
            trigger_storage_max_hold_df.columns = ["n_max_hold"]


            titm_df = titm_df.transpose()
            proz_titm_df = proz_titm_df.transpose()
            VWAP_score_df = VWAP_score_df.transpose()
            pnl_df = pnl_df.transpose()
            trades_df = trades_df.transpose()
            trigger_storage_take_prof_df = trigger_storage_take_prof_df.transpose()
            trigger_storage_stop_loss_df = trigger_storage_stop_loss_df.transpose()
            trigger_storage_max_hold_df = trigger_storage_max_hold_df.transpose()

            global result_per_Share
            result_per_Share = result_per_Share.append([titm_df,proz_titm_df,VWAP_score_df,pnl_df,trades_df,trigger_storage_take_prof_df,
                                                        trigger_storage_stop_loss_df,trigger_storage_max_hold_df])

            global result_matrix
            result_matrix = result_matrix.append(
                {
                    "timestamp": str(timestamp),
                    "exposure": self.market_interface.exposure_total,
                    'pnl': self.market_interface.pnl_realized_total,
                    "n_trades": len(self.market_interface.get_filtered_trades()),
                    "costs": self.market_interface.transaction_cost,
                    "n_orders": len(self.market_interface.get_filtered_orders()),
                    "session_length": self.sess_length,
                    "VMAP_Total_Agent": self.VWAP["Total"],
                }
                , ignore_index=True)

if __name__ == "__main__":

    identifier_list = [
       # ADIDAS
       #"Adidas.BOOK", "Adidas.TRADES",
       # ALLIANZ
       #"Allianz.BOOK", "Allianz.TRADES",
       # BASF
       "BASF.BOOK", "BASF.TRADES",
       # Bayer
       #"Bayer.BOOK", "Bayer.TRADES",
       # BMW
       #"BMW.BOOK", "BMW.TRADES",
       # Continental
       "Continental.BOOK", "Continental.TRADES",
       # Covestro
       #"Covestro.BOOK", "Covestro.TRADES",
       # Daimler
       #"Daimler.BOOK", "Daimler.TRADES",
       # Deutsche Bank
       #"DeutscheBank.BOOK", "DeutscheBank.TRADES",
       # DeutscheBörse
       #"DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    agent = SimpleAgent(
        name="simpleAgent2",
        take_prof=0.002,
        stop_loss=0.002,
        max_holding_period = datetime.timedelta(milliseconds=50, seconds=0 ,minutes=0)
    )
    
    backtest = Backtest(
        agent=agent, 
    )

    """
    # Option 1: run agent against a series of generated episodes, that is, 
    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
        date_start="2016-01-06",
        date_end="2016-01-07",
        episode_interval=30, 
        episode_shuffle=True, 
        episode_buffer=5, 
        episode_length=30, 
        num_episodes=2,
    )
    

    # Option 2: run agent against a series of broadcast episodes, that is, 
    # broadcast the same timestamps for every date between date_start and 
    # date_end
    backtest.run_episode_broadcast(identifier_list=identifier_list,
        date_start="2021-01-04",
        date_end="2021-01-04",
        time_start_buffer="08:00:00", 
        time_start="08:30:00", 
        time_end="09:00:00",
    )
    """

    # Option 3: run agent against a series of specified episodes, that is, 
    # list a tuple (episode_start_buffer, episode_start, episode_end) for each 
    # episode
    backtest.run_episode_list(identifier_list=identifier_list,
        episode_list=episode_list
    )

#print results in excel:
result_per_Share.iloc[:1] = result_per_Share.iloc[:1].astype(str)
result_matrix["session_length"] = result_matrix["session_length"].astype(str)
writer = pd.ExcelWriter('results_benchmark_agent.xlsx')
result_matrix.to_excel(writer,"result_matrix")
result_per_Share.to_excel(writer,"result_per_Share")
writer.save()
