# CorMA: Correlation Matrix Analysis
# Author: Pallab Dutta, Jan 2022
# Work in Progress | This file will be updated with new features
# Thanks to Frank Noe and the DeepTime (https://deeptime-ml.github.io/latest/index.html) 
# documentation for the great representation of the concepts related to VAMP.

import numpy as np
import pandas as pd
import sys
from scipy.linalg import sqrtm

class CorMA:
    def __init__(self,time_series):
        self.time_series = time_series
        self.pca_Cov = None
        self.tic_Cov = None
        self.tcc_Cov = None
    
    def get_cov(self,X,Y,scale=False):
        """
        ----------------- help on get_cov -----------------
        Return depends on scale:
        scale = True   ->  scaled covariance = correlation
        scale = False  ->  covariance
        ---------------------------------------------------
        """
        meanX = np.mean(X,axis=0).reshape(1,-1)
        meanY = np.mean(Y,axis=0).reshape(1,-1)
        N = X.shape[0]
        # covariance matrix
        CovMat = np.matmul((X-meanX).T,(Y-meanY))/N
        if scale:
            stdvX = np.std(X,axis=0).reshape(1,-1)
            stdvY = np.std(Y,axis=0).reshape(1,-1)
            # correlation matrix
            CorMat = CovMat/np.matmul(stdvX.T,stdvY)
            return CorMat
        else:
            return CovMat

    def PCA(self):
        """
        ----------------- Principal Component Analysis -----------------
        Return:
        variance along each principal component (PC) in descending order
        PC corresponding to each variance as column vectors
        ----------------------------------------------------------------
        """
        # Calculate variance-covariance matrix
        CovMat = self.get_cov(self.time_series,self.time_series)
        # Diagonalization
        eigValue,eigVector = np.linalg.eig(CovMat)
        # Sort the eigen vectors with the order of descending eigen values
        idx = np.argsort(eigValue)[::-1]
        eigValue = eigValue[idx]
        eigVector = eigVector[:,idx]
        self.pca_Cov = CovMat
        return eigValue,eigVector

    def TICA(self,lag_time):
        """
        ----------------- Time-lagged Independent Component Analysis -----------------
        Return:
        Correlation along each time independent component (TIC) in descending order
        TIC corresponding to each correlation as column vectors
        ------------------------------------------------------------------------------
        """
        # Calculate diagonalized and whitened covariance matrix
        pcaValue,pcaVector = self.PCA()
        norm_pc = np.matmul(np.linalg.inv(np.diag(pcaValue)),np.matmul(pcaVector,self.time_series))
        # Create time-lagged dataset
        tau1 = norm_pc[:-lag_time,:]
        tau2 = norm_pc[lag_time:,:]
        # Calculate symmetrized time-lagged covariance matrix
        CovMat = 0.5*(self.get_cov(tau1,tau2)+self.get_cov(tau2,tau1).T)
        # Diagonalization
        eigValue,eigVector = np.linalg.eig(CovMat)
        # Sort the eigen vectors with the order of descending eigen values
        idx = np.argsort(eigValue)[::-1]
        eigValue = eigValue[idx]
        eigVector = eigVector[:,idx]
        self.tic_Cov = CovMat
        return eigValue,eigVector

    def TCCA(self,lag_time):
        """
        ----------------- Time-lagged Independent Component Analysis -----------------
        Return:
        Eigenvalue and Eigenvector matrices of the diagonalized time-lagged cannonical 
        correlation matrix
        ------------------------------------------------------------------------------
        """
        # Create time-lagged dataset
        tau1 = self.time_series[:-lag_time,:]
        tau2 = self.time_series[lag_time:,:]
        # Calculate time-lagged correlation matrix
        C00 = self.get_cov(tau1,tau1,scale=True)
        C01 = self.get_cov(tau1,tau2,scale=True)
        C11 = self.get_cov(tau2,tau2,scale=True)
        # Compute time-lagged cannonical correlation matrix
        CovMat = np.matmul(np.linalg.inv(sqrtm(C00)),np.matmul(C01,np.linalg.inv(sqrtm(C11))))
        # Singular value decomposition
        # lsv = left singular vectors
        # rsv = right singular vectors
        # sm = singular matrix
        lsv,sm,rsv = np.linalg.svd(CovMat)
        self.tcc_Cov = CovMat
        return lsv,sm,rsv

    def VAMP_n(self,lag_time,N=2):
        """
        ----------------- Variational Approach to Markov Process Score -----------------
        Return:
        Sum of the diagonal elements raised to the power n, of the diagonalized time-
        lagged cannonical correlation matrix
        --------------------------------------------------------------------------------
        """
        # get singular value decomposed cannonical correlation matrix
        _,Diag,_ = self.TCCA(lag_time=lag_time)
        # extract vamp-n score
        VAMP_n = np.sum(np.power(Diag,N))
        return VAMP_n

def main():
    args = sys.argv[1:]
    t_series = np.load(args[args.index("-trj")+1])
    corranal = CorMA(t_series)
    l_time = int(args[args.index("-dt")+1])
    start = int(args[args.index("-s")+1])
    end = int(args[args.index("-e")+1])+1
    vamp2 = corranal.VAMP_n(lag_time=l_time)
    print("The vamp2 score is: %.3f"%(vamp2))

if __name__ == "__main__":
    main()
