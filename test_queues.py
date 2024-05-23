from queue import PriorityQueue
import pandas as pd 
import heapq

df = pd.DataFrame([1,2])
# # Create a PriorityQueue
priority_queue = PriorityQueue()

# # Add items to the PriorityQueue
priority_queue.put((2, 0, 'task2'))
priority_queue.put((2, 1, 'task0'))
priority_queue.put((1, 0, 'task1'))
priority_queue.put((3, 0, 'task3'))

# # Pop items from the PriorityQueue
# while not priority_queue.empty():
#     priority, task = priority_queue.get()
#     print(f"Processing {task} with priority {priority}")


# heap = []
# heapq.heappush(heap, (1, 'task1'))
# heapq.heappush(heap, (4, df))
# heapq.heappush(heap, (2, 'task2'))
# heapq.heappush(heap, (3, 'task3'))

# while heap:
#     priority, task = heapq.heappop(heap)
#     print(f"Processing {task} with priority {priority}")


heap = []
heapq.heappush(heap, (1, 0, 'b'))
heapq.heappush(heap, (1, 1, 'a'))
heapq.heappush(heap, (2, 0, 'task2'))
heapq.heappush(heap, (3, 0, 'task3'))

while heap:
    priority, p, task = heapq.heappop(heap)
    print(f"Processing {task} with priority {priority}")

print(heap)


