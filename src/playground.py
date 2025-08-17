import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y1 = np.array([92, 87, 95, 93, 88, 89, 95, 92, 93, 96])
y1 = np.array([100, 90,100, 90,100, 90,100, 90,100, 90,])
y2 = np.zeros((10,))

plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.fill_between(x, y1, y2, alpha=0.2, color='green', label='Filled Area')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Filling the Region Between Two Lines')
plt.legend()
plt.show()