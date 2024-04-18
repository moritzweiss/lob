import os 
import matplotlib.pyplot as plt
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from config.config import base, config1 

# print(base['intensities']['market'])
# print(base['intensities']['limit'])
# print(base['intensities']['cancellation'])

plt.figure()
plt.plot(base['intensities']['limit'])
plt.title('base limit')
plt.figure()
plt.plot(base['intensities']['cancellation'])
plt.title('base cancellation')
plt.figure()
plt.plot(base['intensities']['limit']/base['intensities']['cancellation'])
plt.title('base limit/cancellation')


plt.figure()
plt.plot(config1['intensities']['limit'], marker='o')
plt.title('config1 limit')
plt.xlim(0, 12)
plt.xticks(range(0, 13))
plt.grid()
plt.savefig('limits.pdf')
plt.figure()
plt.plot(config1['intensities']['cancellation'], marker='o')
plt.title('config1 cancellation')
plt.xlim(0, 12)
plt.xticks(range(0, 13))
plt.grid()
plt.savefig('cancellations.pdf')
plt.figure()
plt.plot(config1['intensities']['limit']/config1['intensities']['cancellation'], marker='o')
plt.title('config1 limit/cancellation')
plt.xticks(range(0, 13))
plt.xlim(0, 12)
plt.ylim(0,25)
plt.yticks(range(0, 26, 1))
plt.grid()
plt.savefig('ration.pdf')
# plt.xgrid()
plt.show()






