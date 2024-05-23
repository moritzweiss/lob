from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent
from limit_order_book.limit_order_book import LimitOrderBook
import numpy as np
from config.config import noise_agent_config
from queue import PriorityQueue


import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)

# Create a list to be used as a priority queue
# pq = PriorityQueue()
# Add some items to the priority queue
# pq.put(0, PrioritizedItem(2, "Clean the house"))
# pq.put(0, PrioritizedItem(1, "Write code"))
# pq.put(0, PrioritizedItem(3, "Read a book"))

# while not pq.empty():
#     out = pq.get()
#     print(out)



# initialize agents and limit order book 
noise_agent_config['initial_shape_file'] = 'initial_shape/noise_unit.npz'
NA = NoiseAgent(**noise_agent_config)
LSL = LinearSubmitLeaveAgent(when_to_place=0, frequency=100, terminal_time=1000, volume=100)
SA = StrategicAgent(frequency=10, offset=5, market_volume=1, limit_volume=1, rng=np.random.default_rng(0))
LOB = LimitOrderBook(list_of_agents=[NA.agent_id, LSL], level=30, only_volumes=False)


# initialize LOB 
orders = NA.initialize(time=0)
LOB.process_order_list(orders)
# print(LOB.level2('bid'))
# print(LOB.level2('ask'))
# print(LOB.order_map)

# priority queues work like this 
pq = PriorityQueue()

# pq.put((0, 1,'task 1'))
# pq.put((0, 0, 'task 2'))
# pq.put((1, 2, 'task 3'))
# while not pq.empty():
#     priority, _, task = pq.get()
#     print(f"Processing {task} with priority {priority}")

# initial times for all agents 
time = 0 
pq.put(time, 0, 'agent_action')
pq.put(time, 1, 'strategic_agent')
order, waiting_time = NA.sample_order(LOB, time=time)
# should store the noise order somewhere. maybe inside the NA class 
pq.put(time+waiting_time, 0, 'noise_agent')

# 
terminated = False
observation = False
while not terminated and not observation: 
    time, _, task = pq.get()
    if task == 'execution_agent_action':
        # process order 
        orders =LSL.generate_order(time, LOB)
        msgs = LOB.process_order_list(orders)
        LSL.update_position_from_message_list(msgs)
        # set next observation time 
        pq.put(100, 0, 'execution_agent_observation')
    elif task == 'execution_agent_observation':
        observation = True 
        pq.put(time, 0, 'execution_agent_action')
        # exit while loop 
    elif task == 'strategic_agent':
        orders = SA.generate_order(LOB, time=time)
        msgs = LOB.process_order_list(orders)
        LSL.update_position_from_message_list(msgs)
        # new stratetegic agent time  
        # pq.put(, 'strategic_agent')
    elif task == 'noise_agent':
        # 1) process current order at current time 
        # 2) check for terminal condition 
        # 3) generate new order and waiting time 
        order, waiting_time = NA.sample_order(LOB, time=time)
        # save next order in NA class 
        pq.put(time+waiting_time, 2, 'noise_agent')














