import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


class DexVisualizer:
    def __init__(self):
        plt.ion()   # interactive on
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(131)
        self.ay = self.fig.add_subplot(132)
        self.az = self.fig.add_subplot(133)

        self.ax.set_xlabel('sequence')
        self.ax.set_ylabel('offset-x')
        self.ax.set_ylim([-1.5, 1.5])

        self.ay.set_xlabel('sequence')
        self.ay.set_ylabel('offset-y')
        self.ay.set_ylim([-1.5, 1.5])

        self.az.set_xlabel('sequence')
        self.az.set_ylabel('offset-z')
        self.az.set_ylim([0, 2.0])

        self.t = np.linspace(0, 100, 100)
        self.offset_x = np.zeros(len(self.t))
        self.offset_y = np.zeros(len(self.t))
        self.offset_z = np.zeros(len(self.t))

        self.ox, = self.ax.plot(self.t, self.offset_x, 'r-')
        self.oy, = self.ay.plot(self.t, self.offset_y, 'g-')
        self.oz, = self.az.plot(self.t, self.offset_z, 'b-')

        plt.show()

    def update(self, offset):
        self.offset_x = np.append(self.offset_x[1:], np.array(offset[0]))
        self.offset_y = np.append(self.offset_y[1:], np.array(offset[1]))
        self.offset_z = np.append(self.offset_z[1:], np.array(offset[2]))

        self.ox.set_ydata(self.offset_x)
        self.oy.set_ydata(self.offset_y)
        self.oz.set_ydata(self.offset_z)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show_image(self, img, name="Image"):
        cv2.imshow(name, img)
        cv2.waitKey(1)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


if __name__ == "__main__":
    dv = DexVisualizer()

    img = np.random.rand(240, 320)
    img = np.dstack((img, img, img))
    points = np.random.randint(min(img.shape[:2]), size=(10, 2))
    for p in points:
        print(p)
        img = cv2.circle(img, (p[0], p[1]), radius=2, color=(0, 0, 255), thickness=-1)

    dv.show_image(img)
    cv2.waitKey(0)

    # for i in range(100):
    #     offset = np.append(np.random.rand(2) - 0.5, np.random.rand(1))
    #     dv.update(offset)

