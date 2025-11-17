import numpy as np
from scipy.special import erf, erfc


### Expressions for the "mathematical microscope" method
def S0(x, xo, s, a, vtau):
    x = xo - x
    a1 = erfc(x/np.sqrt(2)/s)
    a2 = np.exp(s**2/2/vtau**2 + x/vtau)*erfc(x/np.sqrt(2)/s + s/np.sqrt(2)/vtau)
    return 0.5*a1 - 0.5*(1 - a)*a2


def S(x, xi, wi, s, a, vtau):
    x = xi - x
    return S0(x, xi - wi/2, s, a, vtau) - S0(x, xi + wi/2, s, a, vtau)


def S_(x, xi, wi, s, a, vtau):
    l = 5/3
    return S(x, xi - l, wi, s, a, vtau) + S(x, xi, wi, s, a, vtau) + S(x, xi + l, wi, s, a, vtau)


def math_micro(x, Ig1, Ig2, I1, I2, w1, w2, x1, x2, s, a, vtau):
    l = x2 - x1
    term1 = (1 - np.heaviside(x - (x1 + l/2), 0.5))*Ig1 + (I1 - Ig1)*S_(x, x1, w1, s, a, vtau)
    term2 = np.heaviside(x - (x1 + l/2), 0.5)*Ig2 + (I2 - Ig2)*S_(x, x2, w2, s, a, vtau)
    
    return term1 + term2


def math_micro_5peaks(x, Ig, I1, I2, I3, I4, I5, w1, w2, w3, w4, w5, x1, l1, l2, l3, l4, s, a, vtau):
    l = 5/3

    x1, x2, x3, x4, x5 = np.cumsum([x1, l1, l2, l3, l4])
    
    z = zip([I1, I2, I3, I4, I5], [x1, x2, x3, x4, x5], [w1, w2, w3, w4, w5])

    left = S(x, x1 - l, w1, s, a, vtau)
    right = S(x, x5 + l, w5, s, a, vtau)
    
    return Ig + (I1 - Ig)*left + (I5 - Ig)*right + np.sum(np.array([(Ii - Ig)*S(x, xi, wi, s, a, vtau) for Ii, xi, wi in z]), axis=0)
    

### Expressions for spectral analysis ###

def spectral_10(x, c, a1, b1, l1):

    f1 = 1/l1
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1,], [b1,]), start=1)])
    
    return c + np.mean(yf1, axis=0)


def spectral_20(x, c, a1, a2, b1, b2, l1):

    f1 = 1/l1
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1, a2], [b1, b2]), start=1)])
    
    return c + np.mean(yf1, axis=0)


def spectral_21(x, c, a1, a2, b1, b2, au1, bu1, l1, lu):

    f1, fu = 1/l1, 1/lu
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1, a2], [b1, b2]), start=1)])
    yfu = np.array([aj*np.cos(2*np.pi*j*fu*x) + bj*np.sin(2*np.pi*j*fu*x) for j, (aj, bj) in enumerate(zip([au1,], [bu1,]), start=1)])
    
    return c + np.mean(yf1, axis=0) + np.mean(yfu, axis=0)


def spectral_31(x, c, a1, a2, a3, b1, b2, b3, au1, bu1, l1, lu):

    f1, fu = 1/l1, 1/lu
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1, a2, a3], [b1, b2, b3]), start=1)])
    yfu = np.array([aj*np.cos(2*np.pi*j*fu*x) + bj*np.sin(2*np.pi*j*fu*x) for j, (aj, bj) in enumerate(zip([au1,], [bu1,]), start=1)])
    
    return c + np.mean(yf1, axis=0) + np.mean(yfu, axis=0)
    

def spectral_32(x, c, a1, a2, a3, b1, b2, b3, au1, au2, bu1, bu2, l1, lu):

    f1, fu = 1/l1, 1/lu
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1, a2, a3], [b1, b2, b3]), start=1)])
    yfu = np.array([aj*np.cos(2*np.pi*j*fu*x) + bj*np.sin(2*np.pi*j*fu*x) for j, (aj, bj) in enumerate(zip([au1, au2], [bu1, bu2]), start=1)])
    
    return c + np.mean(yf1, axis=0) + np.mean(yfu, axis=0)


def spectral_33(x, c, a1, a2, a3, b1, b2, b3, au1, au2, au3, bu1, bu2, bu3, l1, lu):

    f1, fu = 1/l1, 1/lu
    yf1 = np.array([aj*np.cos(2*np.pi*j*f1*x) + bj*np.sin(2*np.pi*j*f1*x) for j, (aj, bj) in enumerate(zip([a1, a2, a3], [b1, b2, b3]), start=1)])
    yfu = np.array([aj*np.cos(2*np.pi*j*fu*x) + bj*np.sin(2*np.pi*j*fu*x) for j, (aj, bj) in enumerate(zip([au1, au2, au3], [bu1, bu2, bu3]), start=1)])
    
    return c + np.mean(yf1, axis=0) + np.mean(yfu, axis=0)