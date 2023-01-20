#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 23:17:52 2022
Registration of nuclear locations from two whole slide images
@author: Fayyaz Minhas
"""
from probreg import l2dist_regs
from probreg import callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans
from probreg import cpd
from tqdm import tqdm
import warnings
import pdb
from voronoi import voronoi,pointsInPolygons, getBoundingBox, voronoi_plot_2d
from scipy.special import softmax
import os
#OPENSLIDE_PATH = r"D:\\Dropbox\\PhD_Work\\PythonVE\\openslide-win64-20171122\\bin"
#os.add_dll_directory(OPENSLIDE_PATH)
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.registration.wsi_registration import match_histograms, DFBRegister, apply_bspline_transform, estimate_bspline_transform
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
import cv2
import shutil
from scipy import ndimage
from skimage import color, exposure, measure, morphology

def preprocess_image(image):
    """This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.
    """
    image = color.rgb2gray(image)
    image = exposure.rescale_intensity(
        image, in_range=tuple(np.percentile(image, (0.5, 99.5)))
    )
    image = image * 255
    return image.astype(np.uint8)

class NuclearAligner:
    def __init__(self,M,exemplars = 1000,exemplars_fine = None,removeIsolated = True, filter_knn = 20, filter_p = 95):
        """
        Create a nuclear aligner object for a given fixed (target) point set M

        Parameters
        ----------
        M : TYPE numpy array of size |M|x2
            DESCRIPTION. Target (Fixed) point set
        exemplars : TYPE, optional, the number or target point set exemplars used in coarse registration
            DESCRIPTION. The default is 1000. (determined through kmeans centroids)
        exemplars_fine : TYPE, optional, the numer or target point set exemplars used in fine registration
            DESCRIPTION. The default is None. (set to the same as exemplars)
        filter_knn : TYPE, optional #number of nearest neighbors used in isolated point removal
            DESCRIPTION. The default is 20.
        filter_p : TYPE, optional #filtering percentile
            DESCRIPTION. The default is 95.

        Returns
        -------
        None.

        """
        #self.fine_buffer = fine_buffer
        self.filter_knn = filter_knn
        self.filter_p = filter_p
        if removeIsolated:
            M,_ = self.removeIsolatedPoints(M)
        self.M = M        
        if type(exemplars) == type(0):
            exemplars = self.makeExemplars(M,exemplars)
        self.exemplars = exemplars
        self.setFineExemplars(exemplars_fine)
        
    def setFineExemplars(self,exemplars_fine = None):
        if exemplars_fine is not None:
            if type(exemplars_fine) == type(0):
                exemplars_fine = self.makeExemplars(self.M,exemplars_fine)
            else: self.exemplars_fine = exemplars_fine
        else: exemplars_fine = self.exemplars
        self.exemplars_fine = exemplars_fine
        
    def removeIsolatedPoints(self,M,filter_knn = None, filter_p = None):
        """
        Remove all those points in M whose self.filter_knn neighbor is beyond the self.filter_p percentile of such distances

        Parameters
        ----------
        M : TYPE
            DESCRIPTION.

        Returns
        -------
        M : TYPE
            DESCRIPTION.

        """
        if filter_knn is None: filter_knn = self.filter_knn
        if filter_p is None: filter_p = self.filter_p
        k_d, _ = KDTree(M).query(M, k =  filter_knn)
        thr = np.percentile(k_d[:,-1],filter_p )
        idx = k_d[:,-1]<thr
        M = M[idx]
        return M,idx
    def makeExemplars(self,M,n_exemplars):
        """
        use kmeans to make exemplars

        Parameters
        ----------
        M : TYPE
            DESCRIPTION.
        n_exemplars : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print(M.shape)
        from sklearn.cluster import MiniBatchKMeans as KMeans
        #kM =  KMeans(n_clusters=n_exemplars,batch_size = 2048).fit(M) #
        #kM = M[np.random.permutation(range(M.shape[0]))[:n_exemplars]]#
        kM = M[np.random.choice(M.shape[0], 2, replace=False), :]
        return kM#.cluster_centers_
        
    def preAlign(self,H, exemplars = None, removeIsolated = True):
        """
        perform coarse alignment of H (moving point set) to the fixed point set
        Note: it filters
        
 
        Parameters
        ----------
        H : TYPE moving point set |H|x2
            DESCRIPTION.
        exemplars : TYPE, optional The number of or exemplars of moving point (source) used in coarse alignment
            DESCRIPTION. The default is None. Set to the same number of exemplars as for the target (fixed image)

        Returns
        -------
        transformed points: TYPE
            DESCRIPTION.
        tf_param : TYPE
            DESCRIPTION.

        """
        #M = self.M
        target = self.exemplars
        #filtering to remove isolated points
        #remove all points whose distance to their filter_knn nearest neighbor 
        #is beyond the filter_p percentile of all such distances
        
        if removeIsolated: 
            H,_ = self.removeIsolatedPoints(H)
        if exemplars is None: # if none then we pick the same number of exemplars as used for the target image
            exemplars = self.exemplars.shape[0]        
        if type(exemplars) == type(0): #if number of exemplars is given
            exemplars = self.makeExemplars(H,exemplars)
        source = exemplars #set to given exemplars
        #perform registration
        #cbs = [callbacks.Plot2DCallback(source, target)]
        tf_param, _, _ = cpd.registration_cpd(source, target, 'affine',maxiter=500,tol = 1e-6,use_cuda=False)#,w=0.0)#affine,rigid,nonrigid,callbacks=cbs              
        return tf_param.transform(H), tf_param
    
    def fineAlign(self,H,removeIsolated = True, coarse_transform = None, Nthresh = 1000,fine_buffer = 0): 
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
        M = self.M      
        target_fine = self.exemplars_fine
        
        if coarse_transform is None:
            H2 = H                
        else:
            H2= coarse_transform.transform(H)    
        #kD_target = KDTree(target_fine)        
        #dsource,isource = kD_target.query(H2) #find those points in H2 whose nearest neighbor is a certain target exemplar
        #dtarget,itarget = kD_target.query(M) #find those points in M whose nearest neighbor is a certain target exemplar
                
        H3 = H2*1.0
        Tc = []
        T = []
        bounding_box = getBoundingBox(np.vstack((M,H2)))
        vor = voronoi(target_fine, bounding_box)
        buffered_polygons = [g.buffer(fine_buffer) for g in vor.polygons] #find transform using points in the buffered polygon
        M2poly = pointsInPolygons(M,buffered_polygons)
        H2poly = pointsInPolygons(H2,buffered_polygons)
        H2poly_no_buffer = pointsInPolygons(H2,vor.polygons)
        for idx,region in tqdm(enumerate(vor.polygons)):#tqdm(range(target_fine.shape[0])):
            tidx = M2poly[M2poly.index_polygon == idx].index.tolist() #indices of points within a polygon
            sidx = H2poly[H2poly.index_polygon == idx].index.tolist() #indices of points within a polygon
            #sidx, tidx = isource==idx, itarget==idx
            #pdb.set_trace() #pdb.set_trace = lambda: 1
            Hc,Mc = H2[sidx],M[tidx]
            if len(Hc) and len(Mc):                             
                Hcp = self.makeExemplars(Hc,Nthresh) if Hc.shape[0]>Nthresh else Hc #if too many points, reduce to exemplars
                Mcp = self.makeExemplars(Mc,Nthresh) if Mc.shape[0]>Nthresh else Mc #if too many points, reduce to exemplars
                if removeIsolated:
                    Hcp, _= self.removeIsolatedPoints(Hcp)
                    Mcp, _= self.removeIsolatedPoints(Mcp)
                try: #try selecting exemplars
                    tf_param_c, _, _ = cpd.registration_cpd(Hcp, Mcp, 'rigid',maxiter=200,use_cuda=False)#affine,rigid,nonrigid,callbacks=cbs
                except Exception as e: # in case we get a singular matrix or other error
                    #import pdb; pdb.set_trace() #this happens when there is a single point in the moving image and we are attempting to register it to a bunch of points in the target
                    # that's why it is better to use the image with larger number of nuclei as the moving image (source)
                    warnings.warn(str(e))
                    continue
                T.append(target_fine[idx]) #
                Tc.append(tf_param_c)
                sidx_no_buffer = H2poly_no_buffer[H2poly_no_buffer.index_polygon == idx].index.tolist() #use only points that are in the voronoi cell
                H3[sidx_no_buffer] = tf_param_c.transform(H2[sidx_no_buffer])
        #self.Tc = Tc
        #self.target_fine = np.array(T)
        T = np.array(T)
        fine_transform = (T,Tc)
        return H3,fine_transform,vor
    def transformPoints(self,H,coarse_transform,fine_transform = None,removeIsolated = False):
        """
        DO NOT USE for now
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
        if removeIsolated:
            H, _= self.removeIsolatedPoints(H)
        M = self.M
        H2 = coarse_transform.transform(H)
        if fine_transform is None: return H2
        target_fine,Tc = fine_transform
        target_fine = np.array(target_fine)
        H3 = H2*1.0
        bounding_box = getBoundingBox(np.vstack((M,H2)))
        vor = voronoi(target_fine, bounding_box)       
        H2poly_no_buffer = pointsInPolygons(H2,vor.polygons)        
        
        for idx,region in tqdm(enumerate(vor.polygons)):
            sidx = H2poly_no_buffer[H2poly_no_buffer.index_polygon == idx].index.tolist() #indices of points within a polygon            
            try:
                H3[sidx] = Tc[idx].transform(H2[sidx])
            except KeyError as e: #incase the H points select an exemplar as the nearest which wasn't used 
                warnings.warn(str(e))
                continue
        return H3
def transformPoints_weighted(H,coarse_transform,fine_transform = None,removeIsolated = False,sigma_mul = 0.1): 
    H2 = coarse_transform.transform(H)
    if fine_transform is None: return H2
    target_fine,Tc = fine_transform
    target_fine = np.array(target_fine)
    n_anchors = len(target_fine)
    D = np.zeros((n_anchors,H2.shape[0]))
    Tall = np.zeros((n_anchors,H2.shape[0],H2.shape[1]))
    for c,xc in tqdm(enumerate(target_fine)):
        H3c = Tc[c].transform(H2)
        D[c]=np.linalg.norm(H3c-xc,axis=1)
        Tall[c] = H3c        
    sigma = np.median(D)*sigma_mul
    G = softmax(-D/sigma,axis = 0)
    H3 = np.sum(G[:,:,np.newaxis]*Tall,axis=0)
    return H3

if __name__=='__main__':
    xname,yname = 'Centroid X µm','Centroid Y µm'
    #H = np.array(pd.read_csv(r'B1918044-CDX2.csv',delimiter='\t')[[xname,yname]]) #B1918044_   .sample(100000)
    #M = np.array(pd.read_csv(r'B1928044-HE.csv',delimiter='\t')[[xname,yname]]) #B1918044_
    buffer = 50
    H = np.load("/home/yoelkastro/Desktop/2022-2023/3rd Year Project/fixed_image_land.npy")
    M = np.load("/home/yoelkastro/Desktop/2022-2023/3rd Year Project/moving_image_land.npy")
    
    print(M.shape)
    #if M.shape[0]>H.shape[0]: M,H = H,M 
    NA = NuclearAligner(M = M, exemplars = 20, exemplars_fine = 5, removeIsolated=False)
    
    #%%
    H2,coarse_transform = NA.preAlign(H, removeIsolated=False)
    #%%
    #%load_ext line_profiler
    print("s")
    print(coarse_transform.b)
    print(coarse_transform.t)

    #%lprun -f NA.fineAlign NA.fineAlign(H2)
    #H3,fine_transform,vor = NA.fineAlign(H2,Nthresh = 500, fine_buffer = buffer)
    H3 = H2

    #%%
    from scipy.spatial import Voronoi as voronoi
    from scipy.spatial import voronoi_plot_2d
    #from voronoi import voronoi, voronoi_plot_2d
    fig, axes = plt.subplots(1, 3,  figsize=(7, 6),  sharex=True, sharey=True)
    ax = axes.ravel()
    vor = voronoi(NA.exemplars_fine)
    
    ax[0].scatter(M[:,0],M[:,1],marker = 'o',facecolors='none', edgecolors='r'); ax[0].scatter(H[:,0],H[:,1],marker = '+');# ax[0].scatter(NA.target[:,0],NA.target[:,1], marker = '+',facecolors='g', edgecolors='g');
    ax[0].set_title("Original"); 

    
    ax[1].scatter(M[:,0],M[:,1],marker = 'o',facecolors='none', edgecolors='r'); ax[1].scatter(H2[:,0],H2[:,1],marker = '+');# ax[0].scatter(NA.target[:,0],NA.target[:,1], marker = '+',facecolors='g', edgecolors='g');
    ax[1].set_title("Coarse alignment")
    
    ax[2].scatter(M[:,0],M[:,1],marker = 'o',facecolors='none', edgecolors='r'); ax[2].scatter(H3[:,0],H3[:,1],marker = '+');
    voronoi_plot_2d(vor,ax[2],show_points = False, show_vertices = False, buffer = 50);
    ax[2].set_title("After fine alignment-"+str(buffer))
    
    
    
    
    #%%
    
    #%% ##############################
    
    #1/0
    kM2 = KMeans(n_clusters=100, random_state=0).fit(M)
    kH2 = KMeans(n_clusters=100, init = kM2.cluster_centers_).fit(H2)
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
    #plt.scatter(t_s[:,0],t_s[:,1]); plt.scatter(M[:,0],M[:,1])


    #################################################



    fixed_image_name = 'B1918044_HE.mrxs'
    moving_image_name = 'B-1918044_CDX2p_MUC2y_MUC5g_CD8dab_20220602.mrxs'
    dataset_path = '/home/yoelkastro/Desktop/2022-2023/3rd Year Project/B1918044'

    fixed_img_file_path = os.path.join(dataset_path, fixed_image_name)
    moving_img_file_path = os.path.join(dataset_path, moving_image_name)

    fixed_wsi_reader = WSIReader.open(input_img=fixed_img_file_path)
    moving_wsi_reader = WSIReader.open(input_img=moving_img_file_path)

    thumb_level = 8
    fixed_image = fixed_wsi_reader.slide_thumbnail(resolution=thumb_level, units="level")
    moving_image = moving_wsi_reader.slide_thumbnail(resolution=thumb_level, units="level")
    fixed_mask = fixed_wsi_reader.tissue_mask(resolution=thumb_level, units="level").img
    moving_mask = moving_wsi_reader.tissue_mask(resolution=thumb_level, units="level").img

    # extract tissue region at level 6
    x_fixed, y_fixed, w_fixed, h_fixed = cv2.boundingRect(fixed_mask)
    x_moving, y_moving, w_moving, h_moving = cv2.boundingRect(moving_mask)
    translation_transform_level8 = np.array(
        [
            [1, 0, (x_fixed - x_moving)],
            [0, 1, (y_fixed - y_moving)],
            [0, 0, 1],
        ], dtype=float
    )

    thumb_level = 6      # level 6
    scale_factor = 2**8     # this factor is used to upscale to level 0 from level 8
    orig_fixed_roi = fixed_wsi_reader.read_region((x_fixed*scale_factor, y_fixed*scale_factor), thumb_level, (w_fixed*4, h_fixed*4))
    orig_moving_roi = moving_wsi_reader.read_region((x_moving*scale_factor, y_moving*scale_factor), thumb_level, (w_moving*4, h_moving*4))

    # Preprocessing fixed and moving images
    fixed_roi = preprocess_image(orig_fixed_roi)
    moving_roi = preprocess_image(orig_moving_roi)
    fixed_roi, moving_roi = match_histograms(fixed_roi, moving_roi)

    before_reg_moving = cv2.warpAffine(
        moving_roi, np.eye(2, 3), orig_fixed_roi.shape[:2][::-1]
    )


    newRot = np.ndarray(shape=(2, 3), dtype=float)
    newRot[0] = np.append(coarse_transform.b[0], [coarse_transform.t[0]], axis=0)#coarse_transform.t[0]
    newRot[1] = np.append(coarse_transform.b[1], [coarse_transform.t[1]], axis=0)

    dfbr_registered_image = cv2.warpAffine(
        orig_moving_roi, newRot, orig_fixed_roi.shape[:2][::-1]
    )
    dfbr_registered_mask = cv2.warpAffine(
        moving_mask, newRot, orig_fixed_roi.shape[:2][::-1]
    )

    before_overlay = np.dstack((before_reg_moving, fixed_roi, before_reg_moving))
    dfbr_overlay = np.dstack((dfbr_registered_image[:,:,0], fixed_roi, dfbr_registered_image[:,:,0]))

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(before_overlay, cmap="gray")
    axs[0].set_title("Overlay Before Registration")
    axs[1].imshow(dfbr_overlay, cmap="gray")
    axs[1].set_title("Overlay After Transform")
    plt.show()

    forward_translation = np.array(
        [
            [1, 0, -x_fixed],
            [0, 1, -y_fixed],
            [0, 0, 1],
        ]
    )
    inverse_translation = np.array(
        [
            [1, 0, x_fixed],
            [0, 1, y_fixed],
            [0, 0, 1],
        ]
    )
    dfbr_transform_level8 = newRot[0:-1] * np.array([[1, 1, 1/4], [1, 1, 1/4], [1, 1, 1]])
    image_transform = inverse_translation @ dfbr_transform_level8 @ forward_translation
    final_reg_transform = image_transform @ translation_transform_level8

    registered_image = cv2.warpAffine(
        moving_image, final_reg_transform[0:-1], fixed_image.shape[:2][::-1]
    )

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(fixed_image, cmap="gray")
    axs[0].set_title("Fixed Image")
    axs[1].imshow(registered_image, cmap="gray")
    axs[1].set_title("Registered Image")
    plt.show()