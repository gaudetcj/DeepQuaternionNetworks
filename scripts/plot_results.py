import numpy as np
import matplotlib.pyplot as plt


r = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/real_seg_train_loss.txt')
c = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/complex_seg_train_loss.txt')
q = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/quaternion_seg_train_loss.txt')

plt.plot(r, c='g')
plt.plot(c, c='b')
plt.plot(q, c='r')

r = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/real_seg_val_loss.txt')
c = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/complex_seg_val_loss.txt')
q = np.genfromtxt('d:/Projects/DeepQuaternionNets/scripts/quaternion_seg_val_loss.txt')
print(min(r))
print(min(c))
print(min(q))

plt.plot(r, '--', c='g')
plt.plot(c, '--', c='b')
plt.plot(q, '--', c='r')

plt.title("Kitti Segmentation Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()