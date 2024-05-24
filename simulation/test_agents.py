import sys
import os 
# do not need this once we have https://stackoverflow.com/questions/4757178/how-do-you-set-your-pythonpath-in-an-already-created-virtualenv/47184788#47184788
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
from limit_order_book.limit_order_book import LimitOrder, MarketOrder, CancellationByPriceVolume, Cancellation, LimitOrderBook
from agents import ExecutionAgent, LinearSubmitLeaveAgent

# import numpy as np
# from config.config import config


def test_market_order():
    EA = ExecutionAgent(15, 'ea')
    # EA.reset()
    LOB = LimitOrderBook(['market', 'ea'])
    orders = []
    # finda a better way to set the reference price 
    EA.reference_bid_price = 100
    orders.append(LimitOrder(side='bid', agent_id='market', price=100, volume=10, time=0))
    orders.append(LimitOrder(side='bid', agent_id='market', price=99, volume=10, time=1))
    orders.append(LimitOrder(side='bid', agent_id='market', price=97, volume=10, time=2))
    orders.append(LimitOrder(side='bid', agent_id='market', price=96, volume=10, time=3))
    orders.append(LimitOrder(side='ask', agent_id='market', price=101, volume=10, time=4))
    orders.append(LimitOrder(side='ask', agent_id='market', price=102, volume=20, time=5))
    [LOB.process_order(order) for order in orders]
    # TODO: compact representation of the LOB for printing 
    # print(LOB.level2('bid'))
    # print(LOB.level2('ask'))
    # market sell 
    market_order = MarketOrder(side='bid', agent_id='ea', volume=15, time=6)
    fill = LOB.process_order(market_order)    
    reward, terminated = EA.update_position_from_message_list([fill])
    assert reward == (10*100+5*99 - 15*EA.reference_bid_price)/EA.initial_volume
    assert terminated
    # print(reward)
    market_order = MarketOrder(side='ask', agent_id='ea', volume=15, time=6)
    fill = LOB.process_order(market_order)    
    # market buy: this will throw an error. execution agent is only allowed to sell
    # reward, terminated = EA.update_position_from_message_list([fill])
    return None 


def test_limit_order(terminated=True):
    LOB = LimitOrderBook(['market', 'ea'])
    EA = ExecutionAgent(10, 'ea')
    assert EA.active_volume == 0
    assert EA.volume == 10
    orders = []
    EA.reference_bid_price = 100
    # bid 
    orders.append(LimitOrder(side='bid', agent_id='market', price=100, volume=10, time=0))
    # ask
    orders.append(LimitOrder(side='ask', agent_id='market', price=101, volume=10, time=1))
    orders.append(LimitOrder(side='ask', agent_id='ea', price=101, volume=10, time=2))
    orders.append(LimitOrder(side='ask', agent_id='market', price=102, volume=20, time=3))
    orders.append(LimitOrder(side='ask', agent_id='market', price=103, volume=20, time=4))
    msg_list = [LOB.process_order(order) for order in orders]    
    EA.update_position_from_message_list(msg_list)
    assert EA.active_volume == 10 
    assert EA.volume == 10
    # print(LOB.level2('bid'))
    # print(LOB.level2('ask'))
    if terminated:
        order = MarketOrder(side='ask', agent_id='market', volume=25, time=5)
        fill = LOB.process_order(order)
        # print(LOB.level2('bid'))
        reward, terminated = EA.update_position_from_message_list([fill])
        assert reward == (10*101 - 10*EA.reference_bid_price)/EA.initial_volume
        assert terminated
    else:
        # print(LOB.level2('bid'))
        order = MarketOrder(side='ask', agent_id='market', volume=15, time=5)
        fill = LOB.process_order(order)
        reward, terminated = EA.update_position_from_message_list([fill])
        assert reward == (5*101 - 5*EA.reference_bid_price)/EA.initial_volume
        assert EA.cummulative_reward == reward
        assert not terminated
        assert EA.active_volume == 5
        assert EA.volume == 5
        assert EA.limit_sells == 5        
        assert EA.market_sells == 0
    return None

def sell_remaining():
    LOB = LimitOrderBook(['market', 'ea'])
    EA = ExecutionAgent(10, 'ea')
    EA.reference_bid_price = 100
    # bid 
    orders = []
    orders.append(LimitOrder(side='bid', agent_id='market', price=100, volume=10, time=0))
    # ask
    orders.append(LimitOrder(side='ask', agent_id='market', price=101, volume=10, time=1))
    orders.append(LimitOrder(side='ask', agent_id='ea', price=101, volume=10, time=2))
    orders.append(LimitOrder(side='ask', agent_id='market', price=102, volume=20, time=3))
    orders.append(LimitOrder(side='ask', agent_id='market', price=103, volume=20, time=4))
    msg_list = [LOB.process_order(order) for order in orders]    
    EA.update_position_from_message_list(msg_list)
    # 
    orders = EA.sell_remaining_position(LOB, 5)
    msg_list = [LOB.process_order(order) for order in orders]    
    reward, terminated = EA.update_position_from_message_list(msg_list)
    # 
    assert EA.active_volume == 0
    assert EA.volume == 0
    assert terminated
    assert reward == (10*100 - 10*EA.reference_bid_price)/EA.initial_volume 
    return None

def test_cancellation():
    LOB = LimitOrderBook(['market', 'ea'])
    EA = ExecutionAgent(10, 'ea')
    EA.reference_bid_price = 100
    orders = []
    # bid 
    orders.append(LimitOrder(side='bid', agent_id='market', price=100, volume=10, time=0))
    # ask
    orders.append(LimitOrder(side='ask', agent_id='market', price=101, volume=10, time=1))
    orders.append(LimitOrder(side='ask', agent_id='ea', price=101, volume=10, time=2))
    orders.append(LimitOrder(side='ask', agent_id='market', price=102, volume=20, time=3))
    orders.append(LimitOrder(side='ask', agent_id='market', price=103, volume=20, time=4))
    msg_list = [LOB.process_order(order) for order in orders]    
    EA.update_position_from_message_list(msg_list)
    # 
    assert EA.active_volume == 10
    order = CancellationByPriceVolume(side='ask', agent_id='ea', price=101, volume=10, time=5)
    fill = LOB.process_order(order)
    reward, terminated = EA.update_position_from_message_list([fill])
    # 
    assert EA.active_volume == 0
    assert EA.volume == 10
    # assert terminated
    assert reward == 0
    return None

def test_dynamic():
    LOB = LimitOrderBook(['market', 'ea'])
    EA = ExecutionAgent(9, 'ea')
    EA.reference_bid_price = 100
    orders = []
    # bid 
    orders.append(LimitOrder(side='bid', agent_id='market', price=100, volume=10, time=0))
    # ask
    orders.append(LimitOrder(side='ask', agent_id='market', price=101, volume=1, time=0))
    orders.append(LimitOrder(side='ask', agent_id='ea', price=101, volume=9, time=0))
    msg_list = [LOB.process_order(order) for order in orders]    
    EA.update_position_from_message_list(msg_list)
    # 
    for time in range(1, 11, 1):
        order = MarketOrder(side='ask', agent_id='market', volume=1, time=time)
        msg_list = [LOB.process_order(order)]    
        reward, terminated = EA.update_position_from_message_list(msg_list)
        if terminated:
            break
    assert time == 10
    assert terminated
    assert reward == (101-100)/EA.initial_volume
    assert EA.limit_sells == 9
    assert EA.market_sells == 0
    assert EA.market_buys == 0
    assert (EA.cummulative_reward - 1) < 1e-10
    return None 
    
def test_linear_sl_agent():
    # pass
    volume = 20
    LSL = LinearSubmitLeaveAgent(volume=volume, start_time=0, terminal_time=10, frequency=1)
    LOB = LimitOrderBook(list_of_agents=['noise_agent', LSL.agent_id], level=5, only_volumes=False)
    # initialize the bid 
    order_list = []
    order_list.append(LimitOrder(side='bid', agent_id='noise_agent', price=100, volume=volume, time=0))
    LOB.process_order_list(order_list)
    # agent 
    order = LSL.generate_order(LOB.time, LOB)
    msg = LOB.process_order_list(order)
    LSL.update_position_from_message_list(msg)
    # market order 
    order_list = []
    order_list.append(MarketOrder(side='ask', agent_id='noise_agent', volume=2, time=0))
    msg = LOB.process_order_list(order_list)
    LSL.update_position_from_message_list(msg)
    for time in range(1,11,1):
        # time runs through 1,2, ..., 10
        # establish new bid price cancel and cancel old bid 
        order_list = []
        order_list.append(LimitOrder(side='bid', agent_id='noise_agent', price=100+time, volume=volume, time=time))
        order_list.append(CancellationByPriceVolume(side='bid', agent_id='noise_agent', price=100+time-1, volume=volume, time=time))
        msg = LOB.process_order_list(order_list)
        # l_sl agent sends order 
        order = LSL.generate_order(LOB.time, LOB)
        if order is not None:
            msg = LOB.process_order_list(order)
            LSL.update_position_from_message_list(msg)
        # market order which fills l_sl 
        order_list = []
        order_list.append(MarketOrder(side='ask', agent_id='noise_agent', volume=2, time=time))
        msg = LOB.process_order_list(order_list)
        reward, terminated = LSL.update_position_from_message_list(msg)
        if terminated:
            break
    assert LSL.cummulative_reward == sum([((100+1+time)*2 - 2*100)/20 for time in range(10)])
    pass



test_market_order()
test_limit_order()
test_limit_order(terminated=False)
sell_remaining()
test_cancellation()
test_dynamic()
test_linear_sl_agent()

