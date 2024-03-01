from copy import deepcopy
from operator import neg
from sortedcontainers import SortedDict, SortedList
import numpy as np
import pandas as pd

class MessageDict:
    def __init__(self):
        self.messages = {}
    def add(self, agent_id, message):
        if agent_id in self.messages:
            self.messages[agent_id].append(message)
        else:
            self.messages[agent_id] = [message]

class Order:
    def __init__(self, agent_id, type):
        self.agent_id = agent_id
        self.type = type

class LimitOrder(Order):
    def __init__(self, agent_id, side, price, volume):
        super().__init__(agent_id, 'limit')
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, "volume must be positive"
        self.side = side
        self.price = price
        self.volume = volume
        self.order_id = None
        self.type = 'limit'
    def __repr__(self):
        return f'LO(agent: {self.agent_id}, side: {self.side}, price: {self.price}, volume: {self.volume}, order_id: {self.order_id})'

class MarketOrder(Order):
    def __init__(self, agent_id, side, volume):
        super().__init__(agent_id, 'market')
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, "volume must be positive"
        self.side = side
        self.volume = volume
        self.type = 'market'
    def __repr__(self):
        return f'MO(side: {self.side}, volume: {self.volume})'

class CancellationByPriceVolume(Order):
    def __init__(self, agent_id, side, price, volume):
        super().__init__(agent_id, 'cancellation_by_price_volume')
        assert side in ['bid', 'ask'], "side must be bid or ask"
        assert volume > 0, "volume must be positive"
        self.side = side
        self.price = price
        self.volume = volume
        self.type = 'cancellation_by_price_volume'
    def __repr__(self):
        return f'CBPV(agent: {self.agent_id}, side: {self.side}, price: {self.price}, volume: {self.volume})'

class Cancellation(Order):
    def __init__(self, agent_id, order_id):
        super().__init__(agent_id, 'cancellation')
        assert order_id >= 0, "order id must be positive"
        self.order_id = order_id
        self.agent_id = agent_id
        self.type = 'cancellation'
    def __repr__(self):
        return f'C(order_id: {self.order_id})'

class Modification(Order):
    def __init__(self, agent_id, order_id, volume):
        super().__init__(agent_id, 'modification')
        assert order_id >= 0, "order id must be positive"
        assert volume > 0, "volume must be positive"
        self.order_id = order_id
        self.volume = volume
        self.type = 'modification'
    def __repr__(self):
        return f'M(order_id: {self.order_id}, volume: {self.volume})'

class FillMessage:
    def __init__(self, order, msg):
        self.order = order
        self.msg = msg

class FillMessageLimitOrder(FillMessage):
    def __init__(self, order, msg, order_id):
        super().__init__(order, msg)
        self.order_id = order_id
    def __repr__(self) -> str:
        return f'FLO(msg: {self.msg}, order: {self.order}, order_id: {self.order_id})'


class FillMessageMarketOrder(FillMessage):
    def __init__(self, order, msg, filled_orders, execution_price):
        super().__init__(order, msg)
        self.filled_orders = filled_orders
        self.execution_price = execution_price
    def __repr__(self) -> str:
        return f'FMO(msg: {self.msg}, order: {self.order}, execution_price: {self.execution_price})'


class CancellationMessage(FillMessage):
    def __init__(self, order, msg):
        super().__init__(order, msg)
    def __repr__(self) -> str:
        return f'C(msg: {self.msg}, order: {self.order})'

class CancellationByPriceVolumeMessage(FillMessage):
    def __init__(self, order, msg, filled_orders, filled_volume):
        super().__init__(order, msg)
        self.filled_orders = filled_orders
        self.filled_volume = filled_volume
    def __repr__(self) -> str:
        return f'CBPV(msg: {self.msg}, order: {self.order})'
    
class Data():
    def __init__(self, level) -> None:
        self.level = level 
        self.reset()
    
    def reset(self):
        self.orders = []
        # bid/ask prices and volumes up to a certain level 
        self.bid_prices = []
        self.ask_prices = []
        self.bid_volumes = []
        self.ask_volumes = []
        # best bid/ask prices and volumes
        self.best_bid_prices = []
        self.best_ask_prices = []
        self.best_bid_volumes = []
        self.best_ask_volumes = []
    

class LimitOrderBook:
    def __init__(self, list_of_agents = [], level=10):
        # how many levels of the order book are stored. level=10 means that the first 10 levels of the order book are logged? 
        self.level = level
        self.registered_agents = list_of_agents
        # order ids by agent 
        self.price_map = {'bid': SortedDict(neg), 'ask': SortedDict()}
        self.order_map = {}
        self.order_map_by_agent = {agent_id: set() for agent_id in list_of_agents}
        self.update_n = 0
        # order matters here !!!, first data, then logging 
        self.data = Data(level=level)
        # initialize state of the order book at step n = 0  
        # self._logging()
    
    def _logging(self, order=None):
        # ToDo: increase efficiency of logging
        self.data.orders.append(order)
        # level 2 data including empty levels
        bid_prices, bid_volumes = self.level2('bid')
        ask_prices, ask_volumes = self.level2('ask')
        self.data.bid_prices.append(bid_prices)
        self.data.ask_prices.append(ask_prices)
        self.data.bid_volumes.append(bid_volumes)
        self.data.ask_volumes.append(ask_volumes)
        # best bid/ask prices and volumes
        best_bid_price = self.get_best_price('bid')
        best_ask_price = self.get_best_price('ask')
        best_bid_volume = self.volume_at_price('bid', best_bid_price)
        best_ask_volume = self.volume_at_price('ask', best_ask_price)
        self.data.best_bid_prices.append(best_bid_price)
        self.data.best_ask_prices.append(best_ask_price)
        self.data.best_bid_volumes.append(best_bid_volume)
        self.data.best_ask_volumes.append(best_ask_volume)
    

    def process_order(self, order, log_order=True):
        """
        - an order is a dictionary with fields agent_id, type, side, price, volume, order_id
        - some of those fields are optional depending on the order type 
        """

        # agent_id should be one of the registered agents 
        assert order.agent_id in self.registered_agents, "agent id not registered"

        if order.type == 'limit':
            msg = self.handle_limit_order(order)
        elif order.type == 'market':
            msg = self.handle_market_order(order)
        elif order.type == 'cancellation':
            msg = self.cancellation(order)
        elif order.type == 'modification':
            msg = self.modification(order)
        elif order.type == 'cancellation_by_price_volume':
            msg = self.cancellation_by_price_volume(order)
        else:
            raise ValueError("order type not supported")

        self.update_n += 1 
        # log shape of the book after transition 
        if log_order:        
            self._logging(order)
        else:
            pass
        return msg


    def handle_limit_order(self, order):      
        """
        - if limit price is in price map, add volume to the price level
        - else create a new price level with the volume

        Args:
            order is dict with keys (type, side, price, volume)
        
        Returns:
            None. Changes the state of the order book internally
                - "order_id" is assigned to the order
                - limit order is added to the order map under the key "order_id" and with the whole dict order as value 
                - limit order is added to the price map under the key "price" and with order_id as value
        """

        if order.order_id:
            raise ValueError("limit order should have no order id")
        else:
            order.order_id = self.update_n            

        # only do this check if the opposite side is not empty
        if order.side == 'ask' and self.price_map['bid']:
            assert order.price > self.get_best_price('bid'), "sent ask limit order with price <= bid price"
        if order.side == 'bid' and self.price_map['ask']:    
            assert order.price < self.get_best_price('ask'), "sent bid limit order with price >= ask price"

        if order.price in self.price_map[order.side]:
            # add order to price level 
            self.price_map[order.side][order.price].add(order.order_id) 
        else:
            # SortedList 
            self.price_map[order.side][order.price] = SortedList([order.order_id])
        
        # add order to order map and order map by agent
        self.order_map[order.order_id] = order
        self.order_map_by_agent[order.agent_id].add(order.order_id)

        return FillMessageLimitOrder(order=order, msg='limit order added to the book', order_id=order.order_id)


    def handle_market_order(self, order):
        """
        - match order against limt order in the book
        - return profit message to both agents 
        - modify the state of filled orders in the book 
        """

        side = order.side 
        market_volume = order.volume 

        if not self.price_map[side]:
            raise ValueError(f"{side} side is empty!")

        execution_price = 0.0

        # list of fill messages for each agent         
        filled_orders = {agent_id: [] for agent_id in self.registered_agents}

        prices = list(self.price_map[side].keys())
        for price in prices: 
            # cp = counterparty 
            cp_order_ids = deepcopy(self.price_map[side][price])
            for cp_order_id in cp_order_ids:
                cp_order = self.order_map[cp_order_id]
                cp_agent_id = cp_order.agent_id
                if market_volume < 0:
                    raise ValueError("market volume is negative")
                elif market_volume < cp_order.volume:
                    # counterparty order is partially filled
                    # add additional fields for information, these fields will also be carried in the order map                    
                    cp_order.volume -= market_volume
                    fill_msg = {'partial_fill': True, 'filled_volume': market_volume, 'order': cp_order, 'fill_price': price}
                    filled_orders[cp_agent_id].append(fill_msg)
                    execution_price += price * market_volume
                    market_volume = 0.0
                    # self.order_map[cp_order_id]['volume'] -= market_volume, this is not necessary, opposite order is a reference to opposite_order['volume']
                    break
                elif market_volume >= cp_order.volume:
                    fill_msg = {'partial_fill': False, 'filled_volume': cp_order.volume, 'order': cp_order, 'fill_price': price}
                    filled_orders[cp_agent_id].append(fill_msg)
                    self.price_map[side][price].remove(cp_order_id)              
                    self.order_map.pop(cp_order_id)
                    self.order_map_by_agent[cp_order.agent_id].remove(cp_order.order_id)   # remove is for sets 
                    execution_price += price * cp_order.volume
                    market_volume = market_volume - cp_order.volume
                else:
                    raise ValueError("this should not happen")
            # if no more orders are left on the level, remove the entire price level
            if not self.price_map[side][price]:
                self.price_map[side].pop(price)
            if market_volume == 0.0:
                break

        if market_volume > 0.0: 
            print(f"market order of size {order.volume} not fully executed, {market_volume} remaining!")            
        
        # filled_volume = order.volume - market_volume
        # market: return type, filled orders, average price
        # limit: return order_id
        # cancellation: return ids of cancelled orders          
        return FillMessageMarketOrder(order=order, msg='market order filled', filled_orders=filled_orders, execution_price=execution_price)


    def cancellation(self, order):
        assert order.agent_id in self.order_map_by_agent, "order id not in order map by agent"
        assert order.order_id in self.order_map, "order id not in order map"
        assert order.agent_id == self.order_map[order.order_id].agent_id, "agent id does not match order id"
        # select id, side, price 
        id = order.order_id
        side = self.order_map[order.order_id].side
        price = self.order_map[order.order_id].price
        # remove 
        self.price_map[side][price].remove(id)
        self.order_map.pop(id)
        self.order_map_by_agent[order.agent_id].remove(id)       
        # delete price level if empty
        if not self.price_map[side][price]:
            self.price_map[side].pop(price)
        return CancellationMessage(order=order, msg=f'order with id {id} cancelled')
    
    def cancellation_by_price_volume(self, order):
        assert order.agent_id in self.registered_agents, "agent id not registered"
        assert order.side in ['bid', 'ask'], "side must be either bid or ask"
        assert order.price in self.price_map[order.side], "price not in price map"
        assert order.volume >= 0, "volume must be positive"

        volume = order.volume
        filled_orders = {agent_id: [] for agent_id in self.registered_agents}

        for cp_order_id in self.price_map[order.side][order.price][::-1]:
            if volume == 0:
                break
            cp_order = self.order_map[cp_order_id]
            if cp_order.agent_id == order.agent_id:
                if volume < 0:
                    raise ValueError("cancellation volume is negative")
                elif volume < cp_order.volume:
                    # counterparty order is partially filled
                    fill_msg = {'partial_fill': True, 'filled_volume': volume, 'order': cp_order, 'fill_price': order.price}
                    cp_order.volume -= volume
                    filled_orders[cp_order.agent_id].append(fill_msg)
                    volume = 0.0
                    break
                elif volume >= cp_order.volume:
                    fill_msg = {'partial_fill': False, 'filled_volume': cp_order.volume, 'order': cp_order, 'fill_price': order.price}
                    filled_orders[cp_order.agent_id].append(fill_msg)
                    self.price_map[order.side][order.price].remove(cp_order_id)              
                    self.order_map.pop(cp_order_id)
                    self.order_map_by_agent[cp_order.agent_id].remove(cp_order.order_id)       
                    volume = volume - cp_order.volume
            else:
                pass 
        
        return CancellationByPriceVolumeMessage(order=order, msg=f'worst {order.volume} orders at {order.price} cancelled', filled_orders=filled_orders, filled_volume=order.volume-volume)


    def modification(self, order):
        assert order.volume <= self.order_map[order.order_id].volume, "new volume larger than original order volume"
        assert 0 <= order.volume, "order volume is negative"
        # update volume 
        self.order_map[order.order_id].volume = order.volume
        return {'msg': f'order with id {id} modified', 'order': order, 'volume': order.volume}
    
    def get_best_price(self, side):
        if not self.price_map[side]:
            return np.nan
        else: 
            return self.price_map[side].keys()[0]
    
    def level2(self, side):
        """
        why not lists ? 
        if side == 'bid':
            if side is empty:
                - best bid prices = np.empty(level)
                - best bid volumes = np.empty(level)
            else:
                - best bid prices up to level: [p_1, p_2, ... ,p_level]
                - np array of best bid volumes up to level: [v_1, v_2, ... ,v_level]
                - includes empty price levels
            return (bid_prices, bid_volumes)
        """        
        assert side in ['bid', 'ask'], "side must be either bid or ask"

        if side == 'bid':
            if not self.price_map['ask']:
                if not self.price_map['bid']:
                    return np.empty(self.level)*np.nan, np.empty(self.level)*np.nan
                else:
                    prices = np.arange(self.get_best_price('bid'), self.get_best_price('bid')-self.level, -1)
            else:
                prices = np.arange(self.get_best_price('ask')-1, self.get_best_price('ask')-self.level-1, -1)
        
        if side == 'ask':
            if not self.price_map['bid']:
                if not self.price_map['ask']:
                    return np.empty(self.level)*np.nan, np.empty(self.level)*np.nan 
                else:
                    prices = np.arange(self.get_best_price('ask'), self.get_best_price('ask')+self.level, 1)
            else:
                prices = np.arange(self.get_best_price('bid')+1, self.get_best_price('bid')+self.level+1, 1)

        volumes = []
        for price in prices:
            if price in self.price_map[side]:
                volumes.append(self.volume_at_price(side, price))
            else:
                volumes.append(0)
        
        volumes = np.array(volumes)

        return prices, volumes
    
    def volume_at_price(self, side, price):
        if price not in self.price_map[side]:
            return np.nan 
        else:
            return sum([self.order_map[order_id].volume for order_id in self.price_map[side][price]])
    
    def find_queue_position(self, order_id):        
        # note, it is not entirely clear, what a queue position of an order of lot size > 1 should be
        # we just take the first occurence as its queue position 
        # implement all levels option 
        if order_id not in self.order_map:
            raise ValueError('order_id not found on this side of the book')
        order = self.order_map[order_id]        
        queue_position = 0
        level = self.price_map[order.side][order.price]
        for id in level:
            if id == order_id:
                return queue_position
            queue_position += self.order_map[id].volume
        raise ValueError('order_id not found on this price level')
    
    def log_to_df(self):
        time = np.arange(0, self.update_n)        
        data = {'best_bid_price': self.data.best_bid_prices, 'best_ask_price': self.data.best_ask_prices, 'best_bid_volume': self.data.best_bid_volumes, 'best_ask_volume': self.data.best_ask_volumes}
        # data = pd.DataFrame({'best_bid_volume': self.data.best_bid_volumes, 'best_ask_volume': self.data.best_ask_volumes})
        bid_prices = np.vstack(self.data.bid_prices)
        ask_prices = np.vstack(self.data.ask_prices)
        bid_volumes = np.vstack(self.data.bid_volumes)
        ask_volumes = np.vstack(self.data.ask_volumes)
        for i in range(0, self.level):
            data[f'bid_price_{i}'] = bid_prices[:,i]
            data[f'bid_volume_{i}'] = bid_volumes[:,i]
            data[f'ask_price_{i}'] = ask_prices[:,i]
            data[f'ask_volume_{i}'] = ask_volumes[:,i]
        data = pd.DataFrame.from_dict(data)
        orders = {}
        order_type = ['M' if x.type == 'market' else 'L' if x.type == 'limit' else 'C' if x.type == 'cancellation' else 'PC' if x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_side = [x.side if x.type == 'limit' or x.type == 'market' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_size = [x.volume if x.type == 'limit' or x.type == 'market' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]
        order_price = [x.price if x.type == 'limit' or x.type == 'cancellation_by_price_volume' else np.nan for x in self.data.orders]        
        orders['type'] = order_type
        orders['side'] = order_side
        orders['size'] = order_size
        orders['price'] = order_price        
        orders = pd.DataFrame(orders)
        return data, orders
        


if __name__ == "__main__":
    LOB = LimitOrderBook(smart_agent_id='smart_agent', noise_agent_id='noise_agent')
    lo = LimitOrder('noise_agent', 'bid', 100, 10)
    msg = LOB.process_order(lo)
    print(msg)
    lo = LimitOrder('noise_agent', 'ask', 101, 10)
    msg = LOB.process_order(lo)
    print(msg)
    lo = LimitOrder('noise_agent', 'bid', 99, 10)
    msg = LOB.process_order(lo)
    print(msg)
    p = LOB.find_queue_position(lo.order_id)
    mo = MarketOrder('smart_agent', 'bid', 12)
    msg = LOB.process_order(mo)
    # assert msg['execution_price'] == msg['filled_orders']['noise_agent'][0]['fill_price']*msg['filled_orders']['noise_agent'][0]['filled_volume'] + msg['filled_orders']['noise_agent'][1]['fill_price']*msg['filled_orders']['noise_agent'][1]['filled_volume']    
    print(msg)
    lo = LimitOrder('noise_agent', 'bid', 94, 10)
    msg = LOB.process_order(lo)
    lo = LimitOrder('noise_agent', 'bid', 95, 3)
    msg = LOB.process_order(lo)
    print(LOB.order_map)
    c = Cancellation('noise_agent', 5)
    msg = LOB.process_order(c)
    m = Modification('noise_agent', 4, 5)
    msg = LOB.process_order(m)
    m = Modification('noise_agent', 4, 2)
    msg = LOB.process_order(m)
    out = LOB.level2('bid', level=5)
    print(LOB.price_map)
    print('done')
