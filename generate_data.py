# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:07:00 2018

@author: user
"""

import cv2
import numpy as np
import shutil
import os
import random
from tqdm import tqdm

W = 64
H = 64
W2 = 128
H2 = 128
bg = 255 #white

def PolygonVertex(n_vertex, cx=W/2, cy=H/2, r=W-10, theta0 = 0):
    """
    n_vertex : num of polygon's vertex
    cx : center x
    cy : center y
    r : radius of polygon's circumcircle
    theta0 : first vertex theta
    return vertex pos [n_vertex, 2] 
    """
    dtheta = 2*np.pi/n_vertex
    theta = np.arange(theta0, theta0 + 2*np.pi, dtheta)
    pos = np.zeros((n_vertex, 2))
    pos[:,0] = r*np.cos(theta)
    pos[:,1] = r*np.sin(theta) 
    pos +=  np.array([cx,cy])
    pos = pos.astype(np.int32)
    return pos

#一度高画質で描画してからresizeする
def MakeKanizsaPolygon(n_vertex=3, cx=W, cy=H, polyr=10, cr=10, theta0 = 0, 
                       n_img=0, outfiledir="./data/"):
    """
    mode : line, dot, packman... etc
    return Kanizsa illusion image
    """
  
    #background
    img = np.full((H2, W2, 1), bg, dtype=np.uint8)
    
    mask_pts = PolygonVertex(n_vertex, cx, cy, polyr, theta0)
    #r_pts = PolygonVertex(n_vertex, cx, cy, r, theta0 = np.pi)
    
    #頂点の丸
    for i in range(len(mask_pts)):
        cv2.circle(img, (mask_pts[i,0],mask_pts[i,1]), cr, (0, 0, 0),thickness=-1)
        
    """
    if n_vertex % 2 != 0:
        cv2.polylines(img,[r_pts],True,(0,0,0))
    """

    cv2.fillConvexPoly(img, points = mask_pts, color = bg)
    img2 = img
    img = cv2.resize(img, (H, W))   
    cv2.imwrite(outfiledir+"Kanizsa_X/"+"Kanizsa_X_"+"{0:05d}".format(n_img)+".png", img) 
    
    cv2.polylines(img2,[mask_pts],True,(0,0,0))    
    img2 = cv2.resize(img2, (H, W))
    cv2.imwrite(outfiledir+"Kanizsa_Y/"+"Kanizsa_Y_"+"{0:05d}".format(n_img)+".png", img2)
    #cv2.imshow('', img)
    #cv2.waitKey(0)
    
def MakeKanizsaRandomSquare(n_vertex=4, cr=10, n_img=0, outfiledir="./data/"):
    """
    mode : line, dot, packman... etc
    return Kanizsa illusion image
    """
  
    #background
    img = np.full((H, W, 1), bg, dtype=np.int32)
    mask_pts = np.zeros((n_vertex, 2), dtype=np.int32) 
    mask_pts[0,0] = random.randint(0, 27)
    mask_pts[0,1] = random.randint(0, 27)
    mask_pts[1,0] = random.randint(0, 27)
    mask_pts[1,1] = random.randint(37, H)
    mask_pts[2,0] = random.randint(37, W)
    mask_pts[2,1] = random.randint(37, H)
    mask_pts[3,0] = random.randint(37, W)
    mask_pts[3,1] = random.randint(0, 27) 
    mask_pts.astype(np.int32)
    #mask_pts = np.array( [ [100,100], [100,230], [230, 250], [150,70] ] )
    #mask_pts = mask_pts // 2
    #r_pts = PolygonVertex(n_vertex, cx, cy, r, theta0 = np.pi)
    #頂点の丸
    for i in range(n_vertex):
        cv2.circle(img, (mask_pts[i,0],mask_pts[i,1]), cr, (0, 0, 0),thickness=-1)
        
    """
    if n_vertex % 2 != 0:
        cv2.polylines(img,[r_pts],True,(0,0,0))
    """

    cv2.fillConvexPoly(img, points = mask_pts, color = bg)
    img2 = img
    #img = cv2.resize(img,   (H, W))
    #img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
    cv2.imwrite(outfiledir+"Kanizsa_X/"+"Kanizsa_randsq_X_"+"{0:05d}".format(n_img)+".png", img) 
    
    cv2.polylines(img2,[mask_pts],True,(0,0,0))    
    #img2 = cv2.resize(img2, (H, W))
    #img2 = cv2.resize(img2, None, fx = 0.5, fy = 0.5)
    cv2.imwrite(outfiledir+"Kanizsa_Y/"+"Kanizsa_randsq_Y_"+"{0:05d}".format(n_img)+".png", img2)
    #cv2.imshow('', img)
    #cv2.waitKey(0)

def MakeEhrensteinIllusion(cx=int(W/2), cy=int(H/2),cw=20, ch=10, 
                           line_r=50, theta=0, n_lines=10, theta0=0, n_img=0,
                           outfiledir="./data/"):
    """
    return Ehrenstein illusion lines
    """
    
    #background
    img = np.full((H, W, 1), bg, dtype=np.uint8)
    
    dtheta = 2*np.pi/n_lines
    theta = np.arange(theta0, theta0 + 2*np.pi, dtheta)
    pos = np.zeros((n_lines, 2))
    pos[:,0] = line_r*np.cos(theta)
    pos[:,1] = line_r*np.sin(theta) 
    pos +=  np.array([cx,cy])
    pos = pos.astype(np.int32)
     
    for i in range(len(pos)):
        cv2.line(img, (cx, cy), (pos[i,0], pos[i,1]),0, 1)
    
    cv2.ellipse(img, ((cx,cy), (cw,ch),0), 255, thickness=-1)
    cv2.imwrite(outfiledir+"Ehrenstein_X/"+"Ehrenstein_X_"+"{0:05d}".format(n_img)+".png", img)    
    
    cv2.ellipse(img, ((cx,cy), (cw,ch),0), 0, thickness=1)
    cv2.imwrite(outfiledir+"Ehrenstein_Y/"+"Ehrenstein_Y_"+"{0:05d}".format(n_img)+".png", img)
    
def MakeDataSet(outfiledir="./data/"):

    # Delete the entire directory tree if it exists.
    if os.path.exists(outfiledir):
        shutil.rmtree(outfiledir)  
    
    # Make the directory if it doesn't exist.
    if not os.path.exists(outfiledir):
        os.makedirs(outfiledir)
    
    os.makedirs(outfiledir+"Kanizsa_X/")
    os.makedirs(outfiledir+"Kanizsa_Y/")
    os.makedirs(outfiledir+"Ehrenstein_X/")
    os.makedirs(outfiledir+"Ehrenstein_Y/")
    
    count = 0
    """
    pbar = tqdm(total = 27200)
    for n_vertex in range(3,7):
        dtheta = 2*np.pi/n_vertex
        theta = np.arange(0, dtheta, dtheta*0.25)
        for cx in range(-10,15,5):    
            for cy in range(-10,15,5):
                for polyr in range(25,H-5):      
                    for cr in range(10,20,5):
                        for i in range(len(theta)):                         
                            MakeKanizsaPolygon(n_vertex, W+cx, H+cy, polyr,
                                               cr, theta[i], n_img=count)
                            count+=1
                            pbar.update(1)
    pbar.close()
    
    count = 0
    pbar = tqdm(total = 9375)
    for cx in range(-10,15,5):    
        for cy in range(-10,15,5):            
            for cw in range(10,35,5):         
                for ch in range(10,35,5):      
                    for n_lines in range(5, 20):       
                        MakeEhrensteinIllusion(int(W/2)+cx, int(H/2)+cy, cw,ch,
                                               n_lines=n_lines, n_img=count)
                        count+=1
                        pbar.update(1)
    pbar.close()
    """
    for count in tqdm(range(100)):
        MakeKanizsaRandomSquare(n_img=count)

MakeDataSet()
