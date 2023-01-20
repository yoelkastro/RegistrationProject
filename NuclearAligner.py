#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 23:17:52 2022 '/data-ssd/Share_Fayyaz'
Registration of nuclear locations from two whole slide images
@author: Fayyaz Minhas
"""


from probreg import l2dist_regs
from probreg import callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from probreg import cpd
from tqdm import tqdm
import warnings

class NuclearAligner:
    def __init__(self,filter_knn = 20, filter_p = 95, n_exemplars = 1000,n_exemplars_fine = None):
        self.filter_knn = filter_knn
        self.filter_p = filter_p
        self.n_exemplars = n_exemplars
        if n_exemplars_fine is None:
            n_exemplars_fine = n_exemplars
        self.n_exemplars_fine = n_exemplars_fine
    def preAlign(self,M,H):
        """
        Given a fixed (target set) M and a moving (source set) H, find the 
        coarse affine transformation between the points
        This is done based on registratio of a set of exemplar points
        The exemplar points are selected through kmeans

        Parameters
        ----------
        M : TYPE Nx2 numpy array
            DESCRIPTION. target points
        H : TYPE Mx2 numpy array
            DESCRIPTION. source points

        Returns
        -------
        TYPE Mx2 numpy array
            DESCRIPTION. transformed source points with pre-alignment

        """
        #filtering to remove isolated points
        #remove all points whose distance to their filter_knn nearest neighbor 
        #is beyond the filter_p percentile of all such distances
        k_d, _ = KDTree(H).query(H, k = self.filter_knn)
        thr = np.percentile(k_d[:,-1],self.filter_p )
        H = H[k_d[:,-1]<thr]        
        k_d, _ = KDTree(M).query(M, k =  self.filter_knn)
        thr = np.percentile(k_d[:,-1],self.filter_p )
        M = M[k_d[:,-1]<thr]        
        #get clusters
        kM =  KMeans(n_clusters=self.n_exemplars).fit(M) #M[np.random.permutation(range(M.shape[0]))[:self.n_exemplars]]#
        kH = KMeans(n_clusters=self.n_exemplars).fit(H) #H[np.random.permutation(range(H.shape[0]))[:self.n_exemplars]]# 
        source = kH.cluster_centers_
        target = kM.cluster_centers_
        #perform registration
        #cbs = [callbacks.Plot2DCallback(source, target)]
        tf_param, _, _ = cpd.registration_cpd(source, target, 'affine',maxiter=1000,tol = 1e-6,use_cuda=False)#affine,rigid,nonrigid,callbacks=cbs
        self.preAlignTransform = tf_param
        self.target = target #target exemplar
        self.M = M
        self.H = H
        return self.preAlignTransform.transform(H)
    def fineAlign(self,H = None): 
        """
        Perform fine alignment of the source points H to the target points using piece-wise (discontinous) affine registration
        Basically, identify the group of source and target points that share a common nearest exemplar
        register the two groups using an affine transform
        

        Parameters
        ----------
        H : TYPE Mx2 numpy array of source points, optional (otherwise source points in preAlign are used)
            DESCRIPTION. The default is None.

        Returns
        -------
        H3 : TYPE Mx2 numpy array of transformed points
            DESCRIPTION.

        """
        
        if H is None: H = self.H
        M = self.M
        
        kM =  KMeans(n_clusters=self.n_exemplars_fine).fit(M) #M[np.random.permutation(range(M.shape[0]))[:self.n_exemplars]]#        
        target = kM.cluster_centers_
        H2= self.preAlignTransform.transform(H)
        
        kD_target = KDTree(target)
        
        dsource,isource = kD_target.query(H2)
        dtarget,itarget = kD_target.query(M)        
        H3 = H2*1.0
        Tc = []
        T = []
        for idx in tqdm(range(target.shape[0])):
            sidx, tidx = isource==idx, itarget==idx
            if np.any(sidx) and np.any(tidx):
                Hc,Mc = H2[sidx],M[tidx]
                try:
                    tf_param_c, _, _ = cpd.registration_cpd(Hc, Mc, 'affine',maxiter=1000,tol = 1e-6,use_cuda=False)#affine,rigid,nonrigid,callbacks=cbs
                except Exception as e: # in case we get a singular matrix or other error
                    #import pdb; pdb.set_trace() #this happens when there is a single point in the moving image and we are attempting to register it to a bunch of points in the target
                    # that's why it is better to use the image with larger number of nuclei as the moving image (source)
                    warnings.warn(str(e))
                    continue
                T.append(self.target[idx]) #
                Tc.append(tf_param_c)
                H3[sidx] = tf_param_c.transform(Hc)
        self.Tc = Tc
        self.target = np.array(T)
        self.kD_target = KDTree(self.target)
        return H3
    def transformPoints(self,H):
        """
        Transform points H using the existing set of piece-wise (discontinuous) affine transforms

        Parameters: Same as fineAlign
        ----------
        H : TYPE
            DESCRIPTION.

        Returns
        -------
        H3 : TYPE
            DESCRIPTION.

        """
        H2 = self.preAlignTransform.transform(H)
        H3 = H2*1.0
        dsource,isource = self.kD_target.query(H2)
        for idx in tqdm(np.unique(isource)):
            sidx = isource==idx
            try:
                H3[sidx] = self.Tc[idx].transform(H2[sidx])
            except KeyError as e: #incase the H points select an exemplar as the nearest which wasn't used 
                warnings.warn(str(e))
                continue
        return H3
if __name__=='__main__':
    xname,yname = 'Centroid X µm','Centroid Y µm'
    H = np.array(pd.read_csv(r'B1918044_mihc.txt',delimiter='\t')[[xname,yname]])
    M = np.array(pd.read_csv(r'B1918044_HE.txt',delimiter='\t')[[xname,yname]])
    
    #if M.shape[0]>H.shape[0]: M,H = H,M 
    NA = NuclearAligner(n_exemplars = 1000, n_exemplars_fine = 10)
    H2 = NA.preAlign(M, H)
    H3 = NA.fineAlign()

    #%%
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].scatter(M[:,0],M[:,1],marker = 'o',facecolors='none', edgecolors='r'); ax[0].scatter(H2[:,0],H2[:,1],marker = '+');# ax[0].scatter(NA.target[:,0],NA.target[:,1], marker = '+',facecolors='g', edgecolors='g');
    ax[0].set_title("Coarse alignment")
    
    ax[1].scatter(M[:,0],M[:,1],marker = 'o',facecolors='none', edgecolors='r'); ax[1].scatter(H3[:,0],H3[:,1],marker = '+');
    ax[1].set_title("After fine alignment")

    
    
    #%%
    
    #%% ##############################
    1/0
    kM2 = KMeans(n_clusters=1000, random_state=0).fit(M)
    kH2 = KMeans(n_clusters=1000, init = kM2.cluster_centers_).fit(H2)
    source2 = kH2.cluster_centers_
    target2 = kM2.cluster_centers_
    #source2 = tf_param.transform(source)
    Hstd2 = StandardScaler().fit(source2)
    source3 = Hstd2.transform(source2)
    Mstd2 = StandardScaler().fit(target2)
    target3 = Mstd2.transform(target2)
    #cbs = [callbacks.Plot2DCallback(source3, target3)]
    #tf_param2, _, _ = cpd.registration_cpd(source2, target, 'nonrigid',maxiter=1000,tol = 1e-6,callbacks=cbs,use_cuda=False)#affine,rigid,nonrigid
    
    tf_param2 = l2dist_regs.registration_svr(source3, target3, 'nonrigid',  maxiter = 20, sigma=1.0, delta=0.9, gamma=1.0, nu=0.001, alpha=1.0, beta=0.1, use_estimated_sigma=True)#callbacks=cbs, 
    #print(np.mean(np.linalg.norm((tf_param.transform(source)-target[idx])[:,:2],axis=1)))
    
    plt.show()
    #%%
    
    t_s = Hstd2.inverse_transform(tf_param2.transform(Hstd2.transform(H2)))
    plt.scatter(t_s[:,0],t_s[:,1]); plt.scatter(M[:,0],M[:,1])
