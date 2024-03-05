from limit_order_book import LimitOrderBook, LimitOrder, MarketOrder, Cancellation, Modification, CancellationByPriceVolume
import unittest
import numpy as np

class TestOrderBook(unittest.TestCase):

    def test_cancellation_by_volume(self):
        agents = ['smart_agent', 'noise_agent']
        LOB = LimitOrderBook(list_of_agents=agents, level=10)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5))
        orders.append(LimitOrder('smart_agent', 'ask', 102, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2))
        [LOB.process_order(order) for order in orders]
        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=4, side='ask')
        msg = LOB.process_order(order)
        assert LOB.data.ask_volumes[-1][1] == 5 + 2 + 2 - 4 
        assert LOB.order_map[LOB.price_map['ask'][102][-1]].agent_id == 'smart_agent'

        order = CancellationByPriceVolume(agent_id='noise_agent', price=102, volume=6, side='ask')
        LOB.process_order(order)
        assert LOB.data.ask_volumes[-1][1] == 2
        assert LOB.order_map[LOB.price_map['ask'][102][-1]].agent_id == 'smart_agent'


        orders = []
        orders.append(LimitOrder('smart_agent', 'bid', 99, 3))
        orders.append(LimitOrder('smart_agent', 'bid', 99, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 2))
        orders.append(CancellationByPriceVolume(agent_id='noise_agent', price=99, volume=6, side='bid'))
        [LOB.process_order(order) for order in orders]

        assert LOB.data.bid_volumes[-1][1] == 10+6-6
        assert LOB.order_map[LOB.price_map['bid'][99][-1]].agent_id == 'smart_agent'

        order = CancellationByPriceVolume(agent_id='noise_agent', price=99, volume=20, side='bid')
        LOB.process_order(order)

        assert LOB.data.bid_volumes[-1][1] == 3+1 

        assert LOB.order_map[LOB.price_map['bid'][99][-1]].agent_id == 'smart_agent'


        return None



    def test_logging(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'], level=3)
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10)) #1
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5)) #2
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5)) #3
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10)) #4
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5)) #5
        [LOB.process_order(order) for order in orders]  
        assert len(LOB.data.orders) == 5
        assert len(LOB.data.best_ask_prices) == 5
        assert LOB.data.best_ask_prices == [np.nan, np.nan, 101, 101, 101]
        assert LOB.data.best_bid_prices == [99, 100, 100, 100, 100]
        assert np.all(LOB.data.ask_volumes[-1] == np.array([5, 15, 0]))
        assert np.all(LOB.data.bid_volumes[-1] == np.array([5, 10, 0]))
        assert np.all(LOB.data.ask_prices[-1] == np.array([101, 102, 103]))
        assert np.all(LOB.data.bid_prices[-1] == np.array([100, 99, 98]))
        # 
        order = MarketOrder('smart_agent', 'ask', 5)
        LOB.process_order(order)
        assert np.all(LOB.data.ask_volumes[-1] == np.array([0, 15, 0]))
        return None 



    def test_limit_order_insertion(self):
        LOB = LimitOrderBook(smart_agent_id='smart_agent', noise_agent_id='noise_agent')
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5))
        msg = [LOB.process_order(order) for order in orders]
        [print(m) for m in msg]


    def test_market_order_insertion(self):
        LOB = LimitOrderBook(list_of_agents=['smart_agent', 'noise_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5))
        orders.append(MarketOrder('smart_agent', 'bid', 10))
        msg = [LOB.process_order(order) for order in orders]
        [print(m) for m in msg]


    def test_cancellation(self):
        LOB = LimitOrderBook(smart_agent_id='smart_agent', noise_agent_id='noise_agent')
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 99, 10))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 5))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 10))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 5))
        orders.append(Cancellation())
        msg = [LOB.process_order(order) for order in orders]
        [print(m.msg) for m in msg]        
    

    def test_limit_order_fill(self, partial_fill=False):
        """
        - add some limit orders to the book 
        - add one limit order by smart agent
        - check if the order is filled
        """
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        # add noise agent orders 
        orders = []
        # bid 
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1))
        # ask 
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 105, 3))
        # total ask side = 10
        [LOB.process_order(order) for order in orders]
        # add smart agent order
        if partial_fill:
            msg = LimitOrder('smart_agent', 'ask', 104, 2)
            msg = LOB.process_order(msg)
            order_id = msg.order_id
            order = MarketOrder('noise_agent', 'ask', 8)
            msg = LOB.process_order(order)
            assert msg.passive_fills['smart_agent'][0].partial_fill == True
            assert msg.passive_fills['smart_agent'][0].filled_volume == 1 
            assert LOB.order_map[order_id].volume == 1
            assert order_id in LOB.order_map
            assert order_id in LOB.order_map_by_agent['smart_agent']
        else:
            msg = LimitOrder('smart_agent', 'ask', 104, 2)
            msg = LOB.process_order(msg)
            assert msg.agent_id == 'smart_agent'
            order_id = msg.order_id
            order = MarketOrder('noise_agent', 'ask', 10)
            msg = LOB.process_order(order)
            assert msg.passive_fills['smart_agent'][0].partial_fill == False
            assert msg.passive_fills['smart_agent'][0].filled_volume == 2 
            assert order_id not in LOB.order_map
            assert order_id not in LOB.order_map_by_agent['smart_agent']
        return None 
    

    def test_fill_time(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'ask', 101, 10))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 6))
        orders.append(LimitOrder('smart_agent', 'ask', 103, 2))
        [LOB.process_order(order) for order in orders]
        orders = []
        filled = False
        t = 0 
        while not filled:
            order = MarketOrder('noise_agent', 'ask', 2)
            msg = LOB.process_order(order)
            if 'smart_agent' in msg.passive_fills:
                filled = True
                assert msg.passive_fills['smart_agent'][0].filled_volume == 2
                assert t == 5 + 3 
            t += 1
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'ask', 101, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 1))
        orders.append(LimitOrder('smart_agent', 'ask', 104, 1))
        [LOB.process_order(order) for order in orders]
        orders = []
        filled = False
        t = 0 
        while not filled:
            order = MarketOrder('noise_agent', 'ask', 1)
            msg = LOB.process_order(order)
            if 'smart_agent' in msg.passive_fills:
                filled = True
                assert msg.passive_fills['smart_agent'][0].filled_volume == 1
                assert 104 not in LOB.price_map['ask']
                assert 2 not in LOB.order_map
                assert 2 not in LOB.order_map_by_agent['smart_agent']
                assert t == 2
            t += 1
        # LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        # orders = []
        # orders.append(LimitOrder('noise_agent', 'ask', 101, 1))
        # orders.append(LimitOrder('noise_agent', 'ask', 101, 2))
        # orders.append(LimitOrder('noise_agent', 'ask', 102, 1))
        # orders.append(LimitOrder('noise_agent', 'ask', 104, 3))
        # orders.append(LimitOrder('smart_agent', 'ask', 104, 2))
        # [LOB.process_order(order) for order in orders]
        # orders = []
        # filled = False
        # t = 0 
        # while not filled:
        #     order = MarketOrder('noise_agent', 'ask', 1)
        #     msg = LOB.process_order(order)
        #     if 'smart_agent' in msg.passive_fills:
        #         filled = True
        #         assert msg.passive_fills['smart_agent'][0].filled_volume == 1
        #         assert 104 not in LOB.price_map['ask']
        #         assert 2 not in LOB.order_map
        #         assert 2 not in LOB.order_map_by_agent['smart_agent']
        #         assert t == 2
        return None



    def test_cancellation(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 104, 2))
        [LOB.process_order(order) for order in orders]
        order = LimitOrder('smart_agent', 'ask', 104, 3)
        order = LOB.process_order(order)
        order_id = order.order_id
        order = Cancellation(order_id=order_id, agent_id='smart_agent')
        msg = LOB.process_order(order)
        assert order_id not in LOB.order_map
        assert order_id not in LOB.order_map_by_agent['smart_agent']
        return None 
    

    def test_market_order(self): 
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3))
        orders.append(LimitOrder('noise_agent', 'bid', 97, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 102, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 103, 3))
        [LOB.process_order(order) for order in orders]
        market_order = MarketOrder('smart_agent', 'bid', 5)
        msg = LOB.process_order(market_order)   
        assert market_order.agent_id == 'smart_agent'
        assert 'noise_agent' in msg.passive_fills
        assert msg.execution_price == 100*4+99*1
        assert 100 not in LOB.price_map['bid']
        assert len(LOB.price_map['bid'][99]) == 1
        assert LOB.order_map[LOB.price_map['bid'][99][0]] 
        #####
        market_order = MarketOrder('smart_agent', 'ask', 8)
        msg = LOB.process_order(market_order)
        assert msg.execution_price == 101*5 + 102*2 + 103*1
        assert 101 not in LOB.price_map['ask']
        assert 102 not in LOB.price_map['ask']
        assert len(LOB.price_map['ask'][103]) == 1        
        assert LOB.order_map[LOB.price_map['ask'][103][0]].volume == 2
        assert msg.passive_fills['noise_agent'][-1].partial_fill == True
        return None 

    def test_modification(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3))
        orders.append(LimitOrder('noise_agent', 'ask', 101, 2))
        orders.append(LimitOrder('noise_agent', 'ask', 103, 4))
        msgs = [LOB.process_order(order) for order in orders]
        order_id = msgs[-1].order_id
        assert LOB.order_map[order_id].volume == 4
        order = Modification(order_id=order_id, agent_id='noise_agent', new_volume=1)
        LOB.process_order(order)
        assert LOB.order_map[order_id].volume == 1
        return None

    def test_queue_position(self):
        LOB = LimitOrderBook(list_of_agents=['noise_agent', 'smart_agent'])
        orders = []
        orders.append(LimitOrder('noise_agent', 'bid', 100, 1))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 3))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 2))
        orders.append(LimitOrder('noise_agent', 'bid', 100, 2))
        orders.append(LimitOrder('noise_agent', 'bid', 99, 3))
        [LOB.process_order(order) for order in orders]
        p = LOB.find_queue_position(1)
        assert p == 1
        p = LOB.find_queue_position(3)
        assert p == 6
        return None 
        

if __name__ == '__main__':
    TLOB = TestOrderBook()
    print("##########")
    TLOB.test_cancellation_by_volume()
    print('##########')
    TLOB.test_logging()
    print('##########')
    TLOB.test_fill_time()
    print('##########')
    TLOB.test_limit_order_fill(partial_fill=True)
    TLOB.test_limit_order_fill(partial_fill=False)
    print('##########')
    TLOB.test_cancellation()
    print('##########')
    TLOB.test_market_order()
    print('##########')
    TLOB.test_market_order_insertion()
    print('##########')
    TLOB.test_modification()
    print('##########')
    TLOB.test_queue_position()
