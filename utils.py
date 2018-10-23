import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):
    datarows = []
    with open(fname, 'r') as f:
        f.readline(); # read the header
        for line in f.readlines():
            datarows.append([float(s) for s in line.strip().split(',')])
    print(len(datarows))
    data = np.array(datarows)
    psis = data[:,-6]
    vs = data[:,-5]
    steers = data[:,-1]
    return data, psis, vs, steers

def plot(psis, vs, steers, label=None):
    diffs = np.diff(psis)
    xs = vs[:-1]*steers[:-1]
    keep = np.abs(diffs - diffs.mean()) < 3*diffs.std()
    y = diffs[keep,None]
    x = xs[keep,None]
    if label:
        plt.plot(x, y, '.', label=label)
    else:
        plt.plot(x, y, '.')
    plt.ylabel('diffs of psi')
    plt.xlabel('v * steer');
   
def regress(psis, vs, steers, dt, title=None):
    diffs = np.diff(psis)
    xs = vs[:-1] * steers[:-1] * dt
    keep = np.abs(diffs - diffs.mean()) < 3*diffs.std()
    y = diffs[keep,None]
    x = xs[keep,None]

    X = np.hstack((np.ones_like(x), x))
    w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    lf =  -1/w[-1]
    print('intercept, lf', w[0], lf)
    
    # plot data and best fit line
    xx = np.linspace(x.min(), x.max(), 100)
    yy = w[0] * xx
#     plt.figure()
    label = 'est lf: %.3f' % lf
    if title:
        label = '%s; %s' % (title, label)
    plt.plot(x, y, '.');
    plt.plot(xx, yy, label=label);
    
    plt.ylabel('diffs of psi');
    plt.xlabel('v * steer');