"""
    Python version of Bruno Olshausen's ``showrfs.m``

    See the ``showrfs`` function of this module for details.
"""
import numpy as np
import matplotlib.pyplot as plt

def showrfs(A,bg='min', im=None):
    """
    Parameters
    ----------
    A : array
        Receptive field / basis function array of shape L,M where L is a
        square number which corresponds to the number of pixels, and M is the
        number of basis functions (either square or some power of 2)
    bg : 'min' or 'max'
        Set the 'background' (border between RFs) to be either the maximum or
        the minimum value of the current colormap

    im : AxesImage
        will use im._A as the array into which the basis functions will be
        arranged

    Returns 
    -------
    im : AxesImage 
        created by the plt.imshow() commandwhich contains the RFs / basis
    functions (so you can update them)

    Notes
    -----
    This is Bruno Olshausen's ``showrfs.m`` as pythonized by Paul Ivanov

    Examples
    --------
    Display 128 random 10x10 receptive fields

    >>> import numpy as np
    >>> import showrfs
    >>> im = showrfs.showrfs(np.random.rand(100,128)) 

    Replace with new basis functions
    >>> showrfs.showrfs(np.random.rand(100,128), im=im) 
    """
    L,M=A.shape;
    sz=np.sqrt(L);

    if np.floor(np.sqrt(M))**2 != M:
        m=np.int(np.sqrt(M/2))
        n = M/m
    else:
        n=m=int(np.sqrt(M))

    buf=1; # border around RFs

    # allocate one array that all of the basis functions will be inserted into
    if im is None:
        ar=np.ones((buf+m*(sz+buf),buf+n*(sz+buf)));
        if bg=='min':
          ar*=-1
    else:
        ar = im._A

    k=0;

    for j in xrange(m):
        for i in xrange(n):
            clim=np.max(np.abs(A[:,k])); # rescale basis function for display
            x0,y0 = buf+j*(sz+buf),buf+i*(sz+buf)   # offset for basis function
            sl = np.index_exp[x0:x0+sz,y0:y0+sz]    # slice into array
            ar[sl] =  A[:,k].reshape(sz,sz)/clim;
            k+=1

    im = plt.imshow(ar, vmin=-1,vmax=1)
    plt.axis('off')
    return im
