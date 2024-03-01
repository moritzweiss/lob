import numpy as np
from volume_only import OrderBook

        
def test1():
    # initialize order book 
    LOB = OrderBook()
    # add limit orders on the bid and ask side 
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 10.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 3.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 4.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 102, 'volume': 2.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 104, 'volume': 8.0}
    LOB.process_order(order)
    print(LOB.level2())
    # lob shape on ask side: 101: 7.0, 102: 2.0, 104: 8.0
    # market cash = 7*101 + 2*102 + 3*104 = 1019             
    order = {'agent_id': 'i_agent', 'type': 'market', 'side': 'ask', 'price': None, 'volume': 12.0}
    msg = LOB.process_order(order)
    print(f'average price: {msg[0][0]}, filled volume: {msg[0][1]}')
    print(f'filled orders: {msg[1]}')
    print(f'pfilled orders: {msg[2]}')
    print(LOB.level2())
    print('done')



if __name__ == '__main__':
    Book = OrderBook()
    MarketAgent = MarketAgent(LOB=Book)
    MarketAgent.initialize_book()
    for _ in range(int(1e5)):
        MarketAgent.generate_order()
    out = Book.level2(level=50)
    plt.bar(out['bid'].keys(), out['bid'].values())
    plt.bar(out['ask'].keys(), out['ask'].values())
    plt.show()


def test2():
    LOB = OrderBook()
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 1.0}
    print(LOB.process_order(order))
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 1.0}
    print(LOB.process_order(order))
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 102, 'volume': 1.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 1.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 1.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 102, 'volume': 1.0}
    print(LOB.level2())
    order = {'agent_id': 'agent', 'type': 'market', 'side': 'ask', 'price': None, 'volume': 3.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 10.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 1.0}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 1.0}


def test3():
    # initialize order book 
    LOB = OrderBook()
    # add limit orders on the bid and ask side 
    # 0
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'bid', 'price': 100, 'volume': 10.0}
    out = LOB.process_order(order)
    # 1 
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 3.0}
    LOB.process_order(order)
    # 2 
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 101, 'volume': 4.0}
    LOB.process_order(order)
     # 3
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 102, 'volume': 2.0}
    LOB.process_order(order)
    # 4
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 104, 'volume': 8.0}
    LOB.process_order(order)
    # 5
    order = {'agent_id': 'agent', 'type': 'limit', 'side': 'ask', 'price': 103, 'volume': 2.0}
    LOB.process_order(order)
    print(LOB.level2())
    # 6
    order = {'agent_id': 'smart_agent', 'type': 'limit', 'side': 'ask', 'price': 103, 'volume': 2.0}
    LOB.process_order(order)
    # 7, cancel limit order 
    order = {'agent_id': 'agent', 'type': 'cancellation', 'order_time': 1}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'cancellation', 'order_time': 2}
    LOB.process_order(order)
    order = {'agent_id': 'agent', 'type': 'cancellation', 'order_time': 3}
    LOB.process_order(order)
    print(LOB.level2())
    # 8, cancel market order 
    order = {'agent_id': 'agent', 'type': 'market', 'side': 'ask', 'volume': 5.0}
    market_fill_msg, filled, partially_filled = LOB.process_order(order)
    print(market_fill_msg)
    print(filled)
    print(partially_filled)
    print(1)

def test(): 
    levels = 30
    OB = OrderBook()
    print('initialize order book at ask prices 1001, 1002, ... and bid prices 1000, 999, ... with volume 1')
    for idx, price in enumerate(np.arange(1000, 1000-levels, -1)):
            OB.price_map['bid'].update({price: 1})
    for idx, price in enumerate(np.arange(1001, 1000+levels+1, 1)): 
            OB.price_map['ask'].update({price: 1})    
    print(OB.get_best_volumes('bid', level=levels))
    print(OB.get_best_volumes('ask', level=levels)) 

    print('add 10 market orders to the ask side')
    for _ in range(10):
        OB.process_order({'type': 'market', 'side': 'ask', 'price': None, 'volume': 1})

    print(OB.get_best_volumes('ask', level=levels))
    print(OB.get_best_volumes('bid', level=levels))

    print('add 10 limit orders to the ask side')
    for n in np.arange(1,11,1):
        price = OB.get_best_price('bid')
        OB.process_order({'type': 'limit', 'side': 'ask', 'price': price+n, 'volume': 1})

    print(OB.get_best_volumes('ask', level=levels)) 
    print(OB.get_best_volumes('bid', level=levels))

    print('add 3 market orders to the ask side')
    for n in range(3):
        OB.process_order({'type': 'market', 'side': 'ask', 'price': None, 'volume': 1})
    
    print(OB.get_best_volumes('bid', level=levels))
    print(OB.get_best_volumes('ask', level=levels)) 

    print('add 4 limit orders to the bid side')
    for n in range(1, 5, 1):
        price = OB.get_best_price('ask')
        OB.process_order({'type': 'limit', 'side': 'bid', 'price': price-n, 'volume': 1})

    print(OB.get_best_volumes('bid', level=levels))
    print(OB.get_best_volumes('ask', level=levels)) 

    print('add 4 limit orders to the bid side')
    for n in range(1, 5, 1):
        price = OB.get_best_price('ask')
        OB.process_order({'type': 'limit', 'side': 'bid', 'price': price-n, 'volume': 1})

    print(OB.get_best_volumes('bid', level=levels))
    print(OB.get_best_volumes('ask', level=levels)) 

    print('cancel 4 bid orders')
    for n in range(1, 5, 1):
        price = OB.get_best_price('ask')
        OB.process_order({'type': 'cancellation', 'side': 'bid', 'price': price-n, 'volume': 1})

    print(OB.get_best_volumes('bid', level=levels))
    print(OB.get_best_volumes('ask', level=levels)) 

    print(1)


if __name__ == "__main__":
    test()
