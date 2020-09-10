import numpy as np


def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

def warp_keypoints(data, keypoints, inv=False):
  	
        H = data['homography']
        if 'distortion_factor' in data:
 	        distortion_factor = data['distortion_factor']
 	        distortion_center = data['distortion_center']
 	        
        
        if inv: # if inverse, first apply undistortion, then H_inv
            if 'distortion_factor' in data:
                cen = np.stack([distortion_center[0],distortion_center[1]],0)
                delta = (keypoints - cen)
                norms = np.linalg.norm(delta, ord=None, axis=1 ,keepdims=True)**2#np.linalg.norm(delta)**2 # #np.linalg.norm(delta ,keepdims=True)**2 #might need correction in tests
                denom = distortion_factor*norms+1
                warped_points =( delta / denom )+ cen
            else:
                warped_points = keypoints
            H = np.linalg.inv(H)
            num_points = warped_points.shape[0]
            homogeneous_points = np.concatenate([warped_points, np.ones((num_points, 1))],axis=1)
            warped_points = np.dot(homogeneous_points, np.transpose(H))
            warped_points = warped_points[:, :2] / warped_points[:, 2:]

        else: # if not inverse first apply H then distort
            num_points = keypoints.shape[0]
            homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],axis=1)
            warped_points = np.dot(homogeneous_points, np.transpose(H))
            warped_points = warped_points[:, :2] / warped_points[:, 2:]
 	          
            if 'distortion_factor' in data: 
                cen = np.stack([distortion_center[0],distortion_center[1]],0) 
                delta =  (warped_points - cen)	
                norms = np.linalg.norm(delta, ord=None, axis=1 ,keepdims=True)**2#np.reshape(np.sum(delta**2, axis=1), [delta.shape[0],1])#np.linalg.norm(delta)**2 # #np.linalg.norm(delta ,keepdims=True)**2 #might need correction in tests
                nom = 1-np.sqrt(1-4*distortion_factor*norms)
                denom = 2*distortion_factor*norms
                warped_points = (cen + (nom/denom)*delta)
        
        return (warped_points)
       

def keep_true_keypoints(data, points, inv=False):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        shape = data['prob'].shape
        warped_points = warp_keypoints(data, points[:, [1, 0]], inv)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]
def filter_keypoints(points, shape):
		
		""" Keep only the points whose coordinates are
        inside the dimensions of shape. """
		mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
		 (points[:, 1] >= 0) & (points[:, 1] < shape[1])
		return points[mask, :]