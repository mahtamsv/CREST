import numpy as np
import scipy as sp

from sklearn.covariance import shrunk_covariance

from pyriemann.utils import mean as mr
from pyriemann.utils import distance as dst


def calculate_cp(X1, X2, tORs):
    # calculates the common spatial or temporal filters
    # X1 and X2: inputs for classes 1 and 2 
    # tORS: `time' for CTP or `space' for CSP 
    # X1 and X2 are of the following format: [num_trials, num_channels, time_samps]

    X = np.concatenate((X1,X2),axis=0)
    reg_param = cal_shrinkage(X, tORs)

    c1 = np.mean(cal_covariance(X1, tORs, reg_param),axis=0)
    c2 = np.mean(cal_covariance(X2, tORs, reg_param),axis=0)


    d,v = sp.linalg.eigh(c1, c2)  

    # sort the eigenvalues and -vectors 
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    filts = v.take(indx, axis=1)
    
    return filts



def cal_shrinkage(X, tORs):

    # for X with format: [num_trials, num_channels, time_samps] 
    # calculate the shrinkage parameter based on Target B in the following paper:
    # Schafer and Strimmer, ``A Shrinkage Approach to Large-Scale Covariance Matrix Estimation
    # and Implications for Functional Genomics," 2005.

    if tORs=='time':
        n_Samps = np.size(X, axis=2)
    elif tORs=='space':
        n_Samps = np.size(X, axis=1)


    num_trials = np.size(X, axis=0)

    W = np.zeros([num_trials, n_Samps, n_Samps])

    for ik in range(num_trials):
        if tORs=='time':
            W[ik, :, :] = np.cov(X[ik].transpose())/np.trace(np.cov(X[ik].transpose()))
        elif tORs=='space':
            W[ik, :, :] = np.cov(X[ik])/np.trace(np.cov(X[ik]))

    S = (n_Samps/float(n_Samps-1))*np.mean(W, axis=0)
    v = np.mean(np.diag(S))

    VS = (n_Samps**2/float(n_Samps-1)**3)*np.var(W, axis = 0)

    sh_param = np.sum(VS)/(2*np.sum(np.square(np.triu(S,k=1)))+np.sum(np.square(np.diag(S)-v)))
    return sh_param


def cal_covariance(X, tORs, reg_param):
    # calculate the spatial or temporal covariances for the trials in X
    # X has the following format [num_trials, num_channels, time_samps]

    if tORs=='time':
        n_Samps = np.size(X, axis=2)
    elif tORs=='space':
        n_Samps = np.size(X, axis=1)


    num_trials = np.size(X, axis=0)

    W = np.zeros([num_trials, n_Samps, n_Samps])
    for ik in range(num_trials):
        if tORs=='time':
            temp = np.cov(X[ik].transpose())/np.trace(np.cov(X[ik].transpose()))
        elif tORs=='space':
            temp = np.cov(X[ik])/np.trace(np.cov(X[ik]))
        
        W[ik, :, :] = shrunk_covariance(temp, reg_param)


    return W


def apply_cp(X, filt, tORs, numFilt=None):

    temp = np.shape(filt)
    if numFilt is not None:
        columns = np.concatenate((np.arange(0,numFilt),np.arange(temp[1]-numFilt,temp[1])))

        f = filt[:, columns]
        for ij in range(2*numFilt):
            f[:, ij] = f[:, ij]/np.linalg.norm(f[:, ij])
    else:
        f = filt
        for ij in range(np.size(f,1)):
            f[:, ij] = f[:, ij]/np.linalg.norm(f[:, ij])

    f = np.transpose(f)
    num_trials = np.size(X, axis=0)
    dat = list()

    if tORs=='space':
        for ik in range(num_trials):
            temp = np.matmul(f, X[ik])
            dat.append(temp)
    elif tORs=='time':
        for ik in range(num_trials):
            temp = np.matmul(f, X[ik].transpose())
            dat.append(temp)

    return dat


def cal_riem_means(X1, X2, tORs, rType):

    # find the mean of the covariances in X  
    # two options to use: rType == riem or rType == log-eucl

    X = np.concatenate((X1,X2), axis=0)
    reg_param = cal_shrinkage(X, tORs)

    W1 = cal_covariance(X1, tORs, reg_param)
    W2 = cal_covariance(X2, tORs, reg_param)

    # find the riemannian means of the covariances
    if rType == 'riem':
        C1_mean = mr.mean_riemann(W1)
        C2_mean = mr.mean_riemann(W2)
    elif rType == 'log-eucl':
        C1_mean = mr.mean_logeuclid(W1)
        C2_mean = mr.mean_logeuclid(W2)

    return np.array([C1_mean, C2_mean]), reg_param



def cal_riemann_distance(X, Cmean, rType):

    # find the distance of the covariances in X with the average covariance Cmean 
    # two options to use: rType == riem or rType == log-eucl

    num_trials = np.size(X, axis=0)

    output = np.zeros((num_trials, 1))

    if rType == 'riem':
        for ik in range(num_trials):
            output[ik] = dst.distance_riemann(X[ik], Cmean)
    elif rType == 'log-eucl':
        for ik in range(num_trials):
            output[ik] = dst.distance_logeuclid(X[ik], Cmean)

    return output




