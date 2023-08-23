"""
Basic utilities to keep track of statistics on training, testing etc.

Originally created at 9/17/20, for Python 3.x
2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""
import matplotlib.pylab as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_stats(loss, split, fontsize=12, path=None):
    """Plots figures with loss statistics"""
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel(split, fontsize=fontsize)
    plt.title(split +'ing', fontsize=fontsize)
    if path is not None:
        plt.savefig(path)
        plt.close()

