from advanced import Market
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sortedcontainers import SortedDict, SortedList
from operator import neg

def test_cancellation():
        M = Market()

        # 0 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 3, 'price': 100}
        M.process_order(order)

        # 1
        order = {'agent_id': 'agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 100}
        M.process_order(order)

        # 2 
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 100}
        M.process_order(order)

        # 3
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 2, 'price': 99}
        M.process_order(order)

        # 4
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'bid', 'volume': 1, 'price': 98}
        M.process_order(order)

        # 7
        order = {'agent_id': 'market_agent' , 'type': 'limit', 'side': 'ask', 'volume': 1, 'price': 101}
        M.process_order(order)

        # agent order goes here 
        order = {'agent_id': 'market_agent' , 'type': 'market_cancellation', 'side': 'bid', 'volume': 3, 'price': 100}
        M.find_orders_to_cancel(cancel_volume=3, price=100, side='bid')


        return None

# test_cancellation()



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














