import numpy as np
import cv2
from os import path as osp
from glob import glob

from superpoint.settings import EXPER_PATH
from superpoint.evaluations.evaluation_tool import warp_keypoints, keep_true_keypoints, select_k_best, filter_keypoints


	
def get_paths(exper_name):
    """
    Return a list of paths to the outputs of the experiment.
    """
    return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))


def keep_shared_points(data, keypoint_map, inv, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]   
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(data, keypoints, inv)
    idx = np.argsort(keypoints[:,2])[::-1]
    keypoints = select_k_best(keypoints, keep_k_points)
    return keypoints.astype(int)


def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    shape = data['prob'].shape
    real_H = data['homography']


    # Keeps only the points shared between the two views
    keypoints = keep_shared_points(data, data['prob'], False, keep_k_points)
    warped_keypoints = keep_shared_points(data, data['warped_prob'], True, keep_k_points)
                                         
    desc = data['desc'][keypoints[:, 0], keypoints[:, 1]]
    warped_desc = data['warped_desc'][warped_keypoints[:, 0],
                                      warped_keypoints[:, 1]]

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    if len(matches_idx) == 0:  # No match found
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': [],
                'inliers': [],
                'homography': None}
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv2.RANSAC)
    if H is None:
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': matches,
                'inliers': inliers,
                'homography': H}

    inliers = inliers.flatten()

    # Compute correctness
    corners = np.array([[0, 0, 1],
                        [shape[1] - 1, 0, 1],
                        [0, shape[0] - 1, 1],
                        [shape[1] - 1, shape[0] - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness = float(mean_dist <= correctness_thresh)

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,
            'inliers': inliers,
            'homography': H}


def homography_estimation(exper_name, keep_k_points=1000,
                          correctness_thresh=3, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the correctness score.
    """
    paths = get_paths(exper_name)
    correctness = []
    for path in paths:
        data = np.load(path)
        estimates = compute_homography(data, keep_k_points, correctness_thresh, orb)
        correctness.append(estimates['correctness'])
    return np.mean(correctness)


def get_homography_matches(exper_name, keep_k_points=1000,
                           correctness_thresh=3, num_images=1, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the keypoints shared between the two views,
    a mask of inliers points in the first image, and a list of matches meaning that
    keypoints1[i] is matched with keypoints2[matches[i]]
    """
    paths = get_paths(exper_name)
    outputs = []
    for path in paths[:num_images]:
        data = np.load(path)
        output = compute_homography(data, keep_k_points, correctness_thresh, orb)
        output['image1'] = data['image']
        output['image2'] = data['warped_image']
        outputs.append(output)
    return outputs

def get_ground_truth(data, keypoints, warped_keypoints, shape, correctness_thresh, inv):

    """
    Compute the ground truth keypoints matchings from image to image' where image' in the result 
    of warping image with H_matrix.
    """
    #keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    # Warp the original keypoints with the true homography
    true_warped_keypoints = warp_keypoints(data, keypoints[:, [1, 0]], inv)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                          true_warped_keypoints[:, 0]], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)
    diff = np.expand_dims(warped_keypoints, axis=1) - np.expand_dims(true_warped_keypoints, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, correctness_thresh)
    return matches, len(true_warped_keypoints)

	
def compute_pr_rec(prob, gt, n_gt, total_n, remove_zero=1e-4, simplified=False):
    """
    computes precison and recall of the image
    return: precision and recall
    """ 
    matches = gt
    #print(gt.shape)
    tp = 0
    fp = 0
    tp_points = []
    matched = np.zeros(len(gt))
    for m in matches:
        correct = np.any(m)
        if correct:
            gt_idx = np.argmax(m)
            #tp +=1
            #at most one tp should be considerd for each ground turth point
            if gt_idx not in tp_points:
                tp_points.append(gt_idx)
                tp += 1
            else:
                fp += 1	

        else:
            #tp.append(False)
            fp += 1


    #compute precison and recall
    matching_score = tp / total_n if total_n!=0 else 0
    prec = tp / (tp+fp) if (tp+fp)!=0 else 0
    recall = tp / (n_gt) if n_gt!= 0 else 0
  

    return prec, recall, matching_score
	
def get_mean_AP(data, correctness_threshs, keep_k_points=1000, orb=False):

	prob = data['prob']
	warped_prob = data['warped_prob']


	shape = prob.shape
	warped_shape = warped_prob.shape

	
	# Keeps only the points shared between the two views
	keypoints = keep_shared_points(data, prob, False, 1000)
	warped_keypoints = keep_shared_points(data, warped_prob, True, 1000)



	desc = data['desc'][keypoints[:, 0], keypoints[:, 1]]
	warped_desc = data['warped_desc'][warped_keypoints[:, 0],
										warped_keypoints[:, 1]]

	# Match the keypoints with the warped_keypoints with nearest neighbor search
	if orb:
		desc = desc.astype(np.uint8)
		warped_desc = warped_desc.astype(np.uint8)
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	else:
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		
	matches = bf.match(desc, warped_desc)
	matches_idx = np.array([m.queryIdx for m in matches]).astype(int)
	m_keypoints = keypoints[matches_idx, :]
	matches_idx = np.array([m.trainIdx for m in matches]).astype(int)
	m_warped_keypoints = warped_keypoints[matches_idx, :]
		
	
	precisions = []
	recalls = []
	ms = []

	for t in correctness_threshs:
		#find ground truth
		true_keypoints, n_gt1 = get_ground_truth(data, m_warped_keypoints, m_keypoints, warped_shape, t, inv=True)
		true_warped_keypoints, n_gt2 = get_ground_truth(data, m_keypoints, m_warped_keypoints, shape, t, inv=False)

		#calculate precison and recall 
		prec1, recall1, ms1 = compute_pr_rec(m_warped_keypoints, true_warped_keypoints, n_gt2, len(warped_keypoints))
		prec2, recall2, ms2 =  compute_pr_rec(m_keypoints, true_keypoints, n_gt1, len(keypoints))

		#average precison and recall for two images
		prec = (prec1 + prec2)/2
		recall = (recall1 + recall2)/2
		matching_score = (ms1 + ms2)/2
	
		
		precisions.append(prec)
		recalls.append(recall)
		ms.append(matching_score)

	return precisions, recalls, ms 


def mean_AP(exper_name):
	paths = get_paths(exper_name)
	threshs = np.arange(1,31)
	precisions = np.zeros([1,30])
	recalls = np.zeros([1,30])

	for path in paths:
		data = np.load(path)
		pr, rec, ms = get_mean_AP(data, threshs)
		precisions = np.add(precisions, pr)
		recalls = np.add(recalls, rec)

	n = len(paths)
	precisions = precisions / n
	recalls = recalls / n

	mean_AP = np.sum(precisions[0][1:] * np.abs((np.array(recalls[0][1:]) - np.array(recalls[0][:-1]))))

	return mean_AP

def matching_score(exper_name):
	paths = get_paths(exper_name)
	threshs = np.arange(3,4)
	matching_score = np.zeros([1,1])

	for path in paths:
		data = np.load(path)
		pr, rec, ms = get_mean_AP(data, threshs)
		matching_score = np.add(matching_score, ms)
		
	n = len(paths)	
	matching_score = matching_score/n



	return matching_score
	