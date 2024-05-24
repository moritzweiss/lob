from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent
from limit_order_book.limit_order_book import LimitOrderBook
import numpy as np
from config.config import noise_agent_config
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

# @dataclass(order=True)
# class PrioritizedItem:
#     priority: int
#     item: Any = field(compare=False)

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
noise_agent_config['start_time'] = 0 
NA = NoiseAgent(**noise_agent_config)
EA = LinearSubmitLeaveAgent(start_time=0, time_delta=100, terminal_time=1000, volume=10)
SA = StrategicAgent(start_time=0, time_delta=50, market_volume=1, limit_volume=1, rng=np.random.default_rng(0))
# this reset
SA.reset_direction()
LOB = LimitOrderBook(list_of_agents=[NA.agent_id, EA.agent_id, SA.agent_id], level=30, only_volumes=False)


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


# initialize event queue 
# execution 
out = EA.initial_event()
pq.put(out)
# noise, first time to act 
NA.generate_order(LOB, time=NA.start_time)
out = NA.new_event(NA.start_time, 'noise_agent_action')
pq.put(out)
# strategic. first time to act 
out = SA.initial_event()
pq.put(out)
# should store the noise order somewhere. maybe inside the NA class 

# while not pq.empty():
#     t, p, event = pq.get()
#     print(f"event={event}, time={t}, p={p}")


# 
terminated = False
observation = False
while not terminated and not observation: 
    time, _, event = pq.get()
    # print(f'time={time}')
    # print(f'event={event}')
    if event == 'execution_agent_action':
        orders = EA.generate_order(time, LOB)
        msgs = LOB.process_order_list(orders)
        rewards, terminated = EA.update_position_from_message_list(msgs)
        if terminated:
            break
        else:            
            out = EA.new_event(time, event)
            pq.put(out)
    elif event == 'execution_agent_observation':
        out = EA.new_event(time, event)
        pq.put(out)
        break 
    elif event == 'noise_agent_action':
        msgs = LOB.process_order_list(NA.current_order)
        rewards, terminated = EA.update_position_from_message_list(msgs)
        # safety check
        NA.current_order = None 
        NA.waiting_time = None 
        if terminated:
            break
        else:
            # updates NA.current_order, NA.waiting_time internally 
            NA.generate_order(LOB, time=time)
            out = NA.new_event(time, event)
            pq.put(out)
    elif event == 'strategic_agent_action':
        orders = SA.generate_order(LOB, time=time)
        msgs = LOB.process_order_list(orders)
        rewards, terminated = EA.update_position_from_message_list(msgs)
        if terminated:
            break
        else:
            out = SA.new_event(time, event)
            pq.put(out)
    else:
        raise ValueError(f'event={event} not recognized')

print(f'total reward: {EA.cummulative_reward}')
