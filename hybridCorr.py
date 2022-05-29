import numpy as np
import pandas as pd
from astropy.stats import circcorrcoef
from astropy import units as u
import scipy.stats as ss
import seaborn as sns
import scipy.interpolate

#@title Define Correlation Matrix Function

class matcor:
  """
  Calculates the correlation coefficient among linear-linear, linear-circular and circular-circular variables.
  The coefficient ranges from 0 (No correlation) to 1 (Complete correlation).

  """

  def __init__(self, circ):
    #self.X = X
    #self.Y = Y
    self.circ = circ
    self.N = len(self.circ)
    self.Mcorr = np.zeros([self.N, self.N])

  def corr_LL(self, X, Y):
    crcf = ss.pearsonr(X,Y)[0]
    return abs(crcf)

  def corr_CC(self, X, Y):
    print('corr_CC')
    X=X*u.deg
    Y=Y*u.deg
    crcf = float(circcorrcoef(X, Y))
    return abs(crcf)

  def corr_LC(self, X, Y):
    Y=Y*np.pi/180.0
    sY=np.sin(Y)
    cY=np.cos(Y)
    Rxs = ss.pearsonr(X,sY)[0]
    Rxc = ss.pearsonr(X,cY)[0]
    Rsc = ss.pearsonr(sY,cY)[0]
    crcf = np.sqrt((Rxs**2+Rxc**2-2*Rxs*Rxc*Rsc)/(1-Rsc**2))
    return abs(crcf)

  def ind_corr(self, X, Y, i, j):
    si = self.circ[i]
    sj = self.circ[j]
    iX = X
    jX = Y
    if si==sj and si==0:
      cf = self.corr_LL(iX,jX)
    elif si==sj and si==1:
      cf = self.corr_CC(iX,jX)
    elif si!=sj:
      if si==0:
        cf = self.corr_LC(iX,jX)
      elif sj==0:
        cf = self.corr_LC(jX,iX)
    return cf

  def run(self, X, Y):
    degX=X*180.0/np.pi
    degY=Y*180.0/np.pi
    for i in range(self.N):
      for j in range(i, self.N):
        cf = self.ind_corr(degX[:,i],degY[:,j],i,j)
        self.Mcorr[i,j]=cf
        self.Mcorr[j,i]=cf
    return self.Mcorr
