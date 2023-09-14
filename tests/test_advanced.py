# make sure that imports work 
import sys
import os 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# import modules 
from advanced_multi_lot import Market
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sortedcontainers import SortedDict, SortedList
from operator import neg

# test separately
# - market orders: done 
# - limit orders 
# - cancellations
# - queue position 

# known bug:
# one side of the book is empty: try to calculate level 2 shape 


config = {'total_n_steps': int(1e3), 'log': True, 'seed':0, 'initial_level': 2, 'initial_volume': 100}

def test1():
        """
        - some assertions 
        - but mostly for debugging through the code 
        - check wether queue position is correct
        - check wether orderbook shape is correct 
        """

        id_list = []
        M = Market(config=config)

        # add limit order 
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'bid', 'volume': 3, 'price': 100}
        M.process_order(order)

        # adding a limit order to the book returns the order_id. the order id can be used to cancel the order. 
        # market agent 
        order = {'agent_id': M.agent_id , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        _, id = M.process_order(order)
        id_list.append(id)
        print(f'added {M.order_map[id]} to the book')
        side = M.order_map[id]['side']
        price = M.order_map[id]['price']
        assert price == 100
        assert side == 'bid'
        assert M.price_map[side][price] == [0, 1]
        assert M.find_queue_position(id) == 3 

        # add limit order 
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 100}
        M.process_order(order)

        # add limit order 
        order = {'agent_id': M.agent_id , 'type': 'limit', 'side': 'bid', 'volume': 4, 'price': 100}
        _, id = M.process_order(order)
        id_list.append(id)
        assert M.find_queue_position(id) == 6 

        # limit order on the ask side 
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'ask', 'volume': 4, 'price': 101}
        M.process_order(order)

        # add limit order on the bid side 
        order = {'agent_id': M.agent_id , 'type': 'limit', 'side': 'bid', 'volume': 4, 'price': 100}
        id = M.process_order(order)

        # test shape of the book 
        price, size = M.level1(side='bid')
        assert size == 14 
        price, size = M.level1(side='ask')
        assert size == 4 

        # add limits
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 99}
        M.process_order(order)
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 98}
        M.process_order(order)
        order = {'agent_id': M.market_agent_id , 'type': 'limit', 'side': 'ask', 'volume': 1, 'price': 102}
        M.process_order(order)


        # order book shape 
        M.level2(side='bid', level=2)
        M.level2(side='ask', level=2)

        # order 
        order = {'agent_id': M.agent_id , 'order_id': id_list[0], 'type': 'cancellation'}
        M.process_order(order)

        order = {'agent_id': M.agent_id , 'order_id': id_list[1], 'type': 'cancellation'}
        M.process_order(order)

        M.level2(side='bid', level=2)
        M.level2(side='ask', level=2)

        M.update_n

        order = {'agent_id': M.agent_id , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 100}
        M.process_order(order)

        order = {'agent_id': M.market_agent_id , 'type': 'market', 'side': 'bid', 'volume': 12}
        msg = M.process_order(order)


        return None

test1()


def test_market_order_fill():
    """
    - bid side just 100, ask side 101, 102
    - add 10 lots to the bid side 
    - ask side: 
        - 101: 10 lots market, 3 lots agent, 7 lots market
        - 102: 2 lots agent, 3 lots market
    - send market buy order (ask side) of size 21
    - check wether the cost ist = 2122
    - check whether the first agent order was filled
    - chek wheter the second agent order was partially filled
    - check the level 2 shape on the ask side is [0,4]

    """

    M = Market(config=config)
    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 10, 'price': 100}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 10, 'price': 101}
    M.process_order(order)

    order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'ask', 'volume': 3, 'price': 101}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 7, 'price': 101}
    M.process_order(order)

    order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'ask', 'volume': 2, 'price': 102}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 3, 'price': 102}
    M.process_order(order)
    
    order = {'agent_id': 'agent' , 'type': 'market', 'side': 'ask', 'volume': 21, 'price': None}
    out = M.process_order(order)

    assert out[2] == 2122.0
    assert len(out[1]['agent']) == 2
    order = out[1]['agent'][0] 
    assert order['volume'] == 3
    assert order['order_id'] not in M.order_map.keys()
    order = out[1]['agent'][1]
    assert order['volume'] == 1
    assert order['partial_fill']
    assert order['order_id'] in M.order_map.keys()
    assert (M.level2(side='ask', level=2)[1] == np.array([0, 4])).all()
    return None 


test_market_order_fill()



def test_limit_order_fill():


    for price, time in zip([100, 99, 98], [3, 5, 6]):

        M = Market()

        # 0 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 1
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 2 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 3
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 99}
        M.process_order(order)

        # 4
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 99}
        M.process_order(order)

        # 5
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 98}
        M.process_order(order)

        # 6
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 97}
        M.process_order(order)

        # 7
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 1, 'price': 101}
        M.process_order(order)


        # agent order goes here 
        order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': price}
        M.process_order(order)

        for n in range(10):
            order = {'agent_id': 'market_agent' , 'type': 'market', 'side': 'bid', 'volume': 1, 'price': None}
            out = M.process_order(order)
            if out[1]['agent']:
                print(f'waiting time is {n}')
                print(out[1]['agent'])
                break 

        assert n == time 
    
    return None 

# test_limit_order_fill()


def test_fill_time():

    M = Market()

    # bid side 
    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 100}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 99}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 99}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 98}
    M.process_order(order)

    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 98}
    M.process_order(order)

    # ask side 
    order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 2, 'price': 101}
    M.process_order(order)


    # agent order goes here 
    order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 95}
    M.process_order(order)

    # agent order goes here 
    order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 93}
    M.process_order(order)


    # market 
    # 22, 4, 2, 0, 0, 1, 0, 2   total = 31
    # last limit order will only be partially filled
    order = {'agent_id': 'market_agent' , 'type': 'market', 'side': 'bid', 'volume': 10,'price': None}

    # this fil
    out = M.process_order(order)
    print('filled order is:')
    print(out[1]['agent']) 

    # this function throws an error if one side of the book is empty
    M.plot_level2_order_book()
    
    plt.show()

    return None



def test_limit_order_fill_0():


    for price, m_size in zip([100, 99, 98], [4, 6, 7]):

        M = Market()

        # 0 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 1
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 2 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 3
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 99}
        M.process_order(order)

        # 4
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 99}
        M.process_order(order)

        # 5
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 98}
        M.process_order(order)

        # 6
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 97}
        M.process_order(order)

        # 7
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 1, 'price': 101}
        M.process_order(order)


        # agent order goes here 
        order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': price}
        M.process_order(order)


        order = {'agent_id': 'market_agent' , 'type': 'market', 'side': 'bid', 'volume': m_size, 'price': None}
        out = M.process_order(order)

        if out[1]['agent']:
            # print('market_fills')
            # print(out[1]['market_agent'])
            print('agent fills')
            print(out[1]['agent'])            

    return None 


# test_limit_order_fill_0()


def test_limit_order_fill_1(seed=0):

    for n in range(0, 10):
        M = Market()
        M.np_random  = default_rng(seed)
        M.reset()
        print('#######################')
        print(f'order level {n}')
        print('market reset')
        for order_id in M.order_map:
            order = M.order_map[order_id]
            if order['agent_id'] == 'agent':
                print(order)

        price = 1000 - n
        order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': price}
        out = M.process_order(order)
        for order_id in M.order_map:
            order = M.order_map[order_id]
            if order['agent_id'] == 'agent':
                print(order)


        # generate random market orders
        filled = False 
        n = 0
        m = 0
        while not filled:
            market_volume = M.np_random.lognormal(M.market_volume_parameters['mean'], M.market_volume_parameters['sigma'])
            m += market_volume
            order = {'agent_id': 'market_agent' , 'type': 'market', 'side': 'bid', 'volume': market_volume, 'price': None}
            out = M.process_order(order)
            if out[1]['agent']:
                filled = True
                print(f'cummulative market volume {m}')
                print(out[1]['agent'])
                print(f'fill time {n}')
            n += 1


    return None 


test_limit_order_fill_1()



class TestMarket(Market):
    # super().__init__()

    def reset(self, level=3):
        # super().super().reset(seed=seed)
        # clear book 
        self.order_map = {}
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}

        self.initialize_book()
        best_bid = self.get_best_price(side='bid')
        price = best_bid-level
        # print(f'order submitted at price {price}')
        order = {'agent_id':self.agent_id , 'type':'limit', 'side':'bid', 'price':price, 'volume':1}
        # print(order)
        self.process_order(order)


    
    def step(self, action=0):
        out = self.generate_order()
        return out 
        

def test_cases_fill_time():
    # 3: 8052
    TM = TestMarket()
    TM.np_random = default_rng(0)   
    TM.reset(level=3)
    for n in range(int(1e4)):
        out = TM.step()
        if n == 8051:
            print(TM.level2(side='bid'))
        if n == 8052:
            print(out[-1])
        if out[0] == 'market':
            if out[1]['agent']:
                print(out[1]['agent'])
                print(f'fill time is {n}')


    # 4: 8050. fill time is earlier than level 3 
    # this is because the order book evolves in a slightly different way 
    TM = TestMarket()
    TM.reset(seed=0, level=4)
    for n in range(int(1e4)):
        out = TM.step()
        if n == 8049:
            print(TM.level2(side='bid'))
        if n == 8050:
            print(out[-1])
        if out[0] == 'market':
            if out[1]['agent']:
                print(out[1]['agent'])
                print(f'fill time is {n}')
    
    return None




def test_average_fill_time():
    for level in [2, 3]:
        fill_time = []
        TM = TestMarket()
        TM.np_random = default_rng(0)
        for n in range(75):
            TM.reset(seed=None, level=level)
            for n in range(int(1e4)):
                out = TM.step()
                if out[0] == 'market':
                    if out[1]['agent']:
                        fill_time.append(n)
                        break   
        print(np.mean(fill_time))

# test_average_fill_time()














