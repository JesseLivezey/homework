""" runs Kohonen's self-organizing map algorithm

Python version of Bruno Olshausen's ``kohonen.m`` pythonized by Paul Ivanov
"""
import numpy as np
import matplotlib.pyplot as plt
import showrfs

# input array
imsz=10;
im=np.zeros((imsz,imsz));
imx,imy=np.mgrid[:imsz,:imsz]; # image coordinates

# output array
SZ=10;
X,Y=np.mgrid[:SZ,:SZ];

# width of Gaussian for spreading excitation
sigma=2.5;
decay = .9996

# weights
W=np.random.rand(imsz**2,SZ**2);
W=np.dot(W,np.diag(1./np.sqrt(np.sum(W**2,0))));

# learning rate
eta=0.1;

# init input display
plt.set_cmap('gray')
plt.subplot(221)
h = plt.imshow(im,vmin=0, vmax=1);

t=0;
ax = plt.subplot(222)
plt.subplot(224)
Wim = showrfs.showrfs(W)
for ii in xrange(1000):
    # paint Gaussian blob at a random position in image
    # (you will need to change the procedure for setting x and y
    # to create a scotoma or over-stimulate retina)
    x=imsz*np.random.rand()
    y=imsz*np.random.rand()
    im=np.exp(-0.5*((x-imx)**2+(y-imy)**2));
    #im[2:5,2:5] = 0
    #h.set_data(im), plt.draw()# comment out this line to go faster

    # compute output and find winner
    output = im.ravel().dot(W).reshape(SZ,SZ)

    # spread activation to neighbors of winner
    idx = np.unravel_index(np.argmax(output), output.shape)
    winner = np.zeros_like(output)
    for xx in xrange(SZ):
        for yy in xrange(SZ):
            if (xx-idx[0])**2+(yy-idx[1])**2 <= sigma**2:
                winner[xx, yy] = 1.
    winner = winner.ravel()
    
    # Hebbian weight update
    W += eta*(im.ravel()[np.newaxis,:]-winner[:,np.newaxis]*W)*winner[:,np.newaxis]

    #W=W*diag(1./sqrt(sum(W.^2))); # normalize weight vectors
    W=np.dot(W,np.diag(1./np.sqrt(np.sum(W**2,0))));
    eta = decay*eta
    sigma = decay*sigma

    # display network state every 100 trials
    if t%100==0:
        mux=np.abs(np.dot(W.T, imx.ravel())/np.sum(np.abs(W),0).T)
        mux.resize((SZ,SZ))
        muy=np.abs(np.dot(W.T, imy.ravel())/np.sum(np.abs(W),0).T)
        muy.resize((SZ,SZ))

        ax.plot(mux,muy,'k')
        ax.hold(True)
        ax.plot(mux.T,muy.T,'k')
        ax.hold(False)
        ax.axis([0,SZ-1,0,SZ-1])

        showrfs.showrfs(W,im=Wim)
        plt.draw()

    t=t+1;
plt.subplot(221)
h = plt.imshow(im,vmin=0, vmax=1);
plt.show()
