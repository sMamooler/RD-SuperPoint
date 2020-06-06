import tensorflow as tf
from tensorflow.contrib.image import transform as H_transform
import cv2 as cv
	
from superpoint.utils.tools import dict_update
from superpoint.models.homographies import (homography_adaptation_default_config,sample_homography, 
											invert_homography)


def distortion_homography_adaptation(image, net, config):
	"""Performs radial distortion and homography adaptation.
	Arguments:
		image: a 'Tensor' with shape '[N,H,W,1]'.
		net: A function that takes an image as input, performs inference, and outputs the 
			prediction dictionary.
		config: A configuration dictionary containing the distortion factor 'dist_fact' and optional enteries such as number 
			of sampled homographies 'num', the aggregation method 'aggregation.
	Returns:
		A dictionary which contains the aggregated detection probabilities.
	"""
	probs = net(image)['prob']
	counts = tf.ones_like(probs)
	images = image
	
	probs = tf.expand_dims(probs, axis=-1)
	counts = tf.expand_dims(counts, axis=-1)
	images = tf.expand_dims(images, axis=-1)
	
	shape = tf.shape(image)[1:3]
	config = dict_update(homography_adaptation_default_config, config)
	
	def step(i, probs, counts, images):
		#Sample image patch
		H = sample_homography(shape, **config['homographies'])
		H_inv = invert_homography(H)
		
		#############################################
		H_ = shape[0]
		W = shape[1]
		row_c = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(H_, tf.float32), dtype=tf.float32)
		col_c = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(W, tf.float32), dtype=tf.float32) 
		lambda_ = tf.constant(0.00001)
		#############################################
		#apply the homography 
		warped = H_transform(image, H, interpolation='BILINEAR')
		#############################################
		#apply the radial distortion
		warped = distort(warped,lambda_,(row_c,col_c))
		
		#count = warp_points_dist(tf.expand_dims(tf.ones(tf.shape(image)[:3]),-1), lambda_, (row_c,col_c), inverse=True)
		count = undistort(tf.expand_dims(tf.ones(tf.shape(image)[:3]),-1),lambda_, (row_c, col_c))
		count = tf.round(count)
		count = H_transform(count,H_inv, interpolation='NEAREST')
		
		mask = H_transform(tf.expand_dims(tf.ones(tf.shape(image)[:3]), -1),
							 H,interpolation='NEAREST')
		#mask = warp_points_dist(mask, lambda_, (row_c,col_c), inverse=False)
		mask = distort(mask, lambda_, (row_c,col_c))
		mask = tf.round(mask)
		#############################################
		
		# Ignore the detections too close to the border to avoid artifacts
		if config['valid_border_margin']:
			kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (config['valid_border_margin'] * 2,) * 2)
			with tf.device('/cpu:0'):
				count = tf.nn.erosion2d(
                    count, tf.to_float(tf.constant(kernel)[..., tf.newaxis]),
                    [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[..., 0] + 1.
				mask = tf.nn.erosion2d(
                    mask, tf.to_float(tf.constant(kernel)[..., tf.newaxis]),
                    [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[..., 0] + 1.

        # Predict detection probabilities
		prob = net(warped)['prob']
		prob = prob * mask
		prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv,
                                interpolation='BILINEAR')[..., 0]
		prob_proj = prob_proj * count
		probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
		counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
		images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
		return i + 1, probs, counts, images
	
	_, probs, counts, images = tf.while_loop(
            lambda i, p, c, im: tf.less(i, config['num'] - 1),
            step,
            [0, probs, counts, images],
            parallel_iterations=1,
            back_prop=False,
            shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None, None, None, None]),
                    tf.TensorShape([None, None, None, None]),
                    tf.TensorShape([None, None, None, 1, None])])

	counts = tf.reduce_sum(counts, axis=-1)
	max_prob = tf.reduce_max(probs, axis=-1)
	mean_prob = tf.reduce_sum(probs, axis=-1) / counts

	if config['aggregation'] == 'max':
		prob = max_prob
	elif config['aggregation'] == 'sum':
		prob = mean_prob
	else:
		raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

	if config['filter_counts']:
		prob = tf.where(tf.greater_equal(counts, config['filter_counts']),
                        prob, tf.zeros_like(prob))

	return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug

	
	

	
def distort(image, distortion_factor, distortion_center):
	"""
	Distorts the given image and the distortion factor lambda specified in config
	with a random distortion center.
	
	Arguments: 
			image: a Tensor of shape [N,H,W,1] where N is the numebr of images in 
					the patch, H is the image's height and W is its width.
			distortion_factor: the value that indicates the intensity of distortion.
			distortion_center: the pair (row_c, col_c) that is the coordinate of the distortion 
								center.
	Returns: A tensor of the same type and shape as image which contains the pixel values
			of the distorted version of image, as well as distortion parameters
	"""
	
	shape = tf.shape(image)
	N = shape[0]
	H = shape[1]
	W = shape[2]
	
	d_coords = tf.zeros([H,W,2])
	row_c, col_c = distortion_center
	lambda_ = distortion_factor
	d_image = tf.zeros(shape)

	cols = tf.range(W)
	cols = tf.reshape(cols, [1,W])
	#cols = tf.repeat(cols,H,axis = 0)
	cols = tf.tile(cols, [H,1])
	
	
	rows = tf.range(H)
	rows = tf.reshape(rows, [H,1])
	#rows = tf.repeat(rows, W, axis=1)
	rows = tf.tile(rows, [1,W])
	
	d_coords = tf.stack([cols,rows],axis = 2)
	d_coords = tf.cast(tf.reshape(d_coords, [H,W,2]),tf.float32)
	
	cen = tf.stack([row_c,col_c],0)
	
	delta = tf.cast(d_coords-cen, tf.float32)
	norms = tf.norm(delta,'euclidean',2 ,keepdims=True)**2
	denom = lambda_ * norms + 1
	u_coord = delta / denom + cen
	
	
		
	#image = tf.reshape(image, [1,H,W,1])
	u_coord = tf.reshape(u_coord, [1,H,W,2])
	u_coord = tf.tile(u_coord,[N,1,1,1])
	d_image = tf.contrib.resampler.resampler(image, u_coord)
	#d_image = tf.reshape(d_image,[H,W,1])
	
	return d_image
	
	#return {'distorted_image':d_image,'lambda': lambda_, 'dist_center':[row_c, col_c]}




def undistort(d_image, distortion_factor, distortion_center):
	"""
	Undistorts the given distorted image using the given distortion factor lambda_
	and the distortion center coordinate row_c and col_c
	Arguments: 
			d_image: a Tensor of shape [N,H,W,1] where N is the numebr of images in 
			the patch, H is the image's height and W is its width.
			distortion_factor: the value that indicates the intensity of distortion.
			distortion_center: the pair (row_c, col_c) that is the coordinate of the distortion 
								center.
	
	Returns: A tensor of the same type and shape as d_image which contains the pixel values
			of the undistorted version of d_image.
	"""
	shape = tf.shape(d_image)
	N = shape[0]
	H = shape[1]
	W = shape[2]

	
	row_c, col_c = distortion_center
	lambda_ = distortion_factor
	
	u_coords = tf.zeros([H,W,2])
	
	cols = tf.range(W)
	cols = tf.reshape(cols, [1,W])
	#cols = tf.repeat(cols,H,axis = 0)
	cols = tf.tile(cols, [H,1])
	
	rows = tf.range(H)
	rows = tf.reshape(rows, [H,1])
	#rows = tf.repeat(rows, W, axis=1)
	rows = tf.tile(rows, [1,W])
	
	u_coords = tf.stack([cols,rows],axis = 2)
	u_coords = tf.reshape(u_coords, [H,W,2])
	
	
	delta = tf.cast(u_coords-[row_c, col_c], tf.float32)
	norms = tf.norm(delta,'euclidean',2 ,keepdims=True)**2
	nom = 1-tf.math.sqrt(1-4*lambda_*norms)
	denom = 2*lambda_*norms
	d_coords = [row_c, col_c] + (nom/denom)*delta
	
	

	#d_image = tf.reshape(d_image, [1,H,W,1])
	d_coords = tf.reshape(d_coords, [1,H,W,2])
	d_coords = tf.tile(d_coords,[N,1,1,1])
	u_image = tf.contrib.resampler.resampler(tf.cast(d_image, tf.float32), d_coords)
	#u_image = tf.reshape(u_image,[H,W,1])
	

	return u_image
	
def compute_valid_mask_dist(image_shape, distortion_factor, distortion_center, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from the radial distortion function applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
		config: configuration containing the distortion factor and the radius of the margin to be discarded.
       

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    mask = distort(tf.ones(image_shape)[tf.newaxis,...,tf.newaxis],distortion_factor, distortion_center)
    if erosion_radius > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        mask = tf.nn.erosion2d(
                mask,
                tf.to_float(tf.constant(kernel)[..., tf.newaxis]),
                [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[0, ..., 0] + 1.
    return tf.to_int32(tf.math.round(mask))



def warp_points_dist(points, dist_factor, dist_center, inverse=False):
	"""
	Warp a list of points with either the inverse of the radial distortion
	or the radial distortion intself depending on the value of 'inverse'
	
	Arguments:
		points: list of N points, shape (N,2).
		dist_factor: a double, the distortion factor used to distort the image.
		dist_center: a pair, the distortion center used to distort the image.
		inverse: a boolean that indicates whether 'points' are keypoints of the distorted image,
				or the original image. If inverse=True, they are keypoints of the
				distorted image, o.w of the original image
		
	Returns: a Tensor of shape (N,2) containing the new coordinates of the 
				warped points.
	
	"""
	points = tf.cast(points, tf.float32)
	delta = points - [dist_center[0],dist_center[1]]
	norms = tf.norm(delta,'euclidean',1 ,keepdims=True)**2
	
	if inverse:
		denom = dist_factor * norms + 1
		warped_points = delta / denom + dist_center
	else:	
		nom = 1-tf.math.sqrt(1-4*dist_factor*norms)
		denom = 2*dist_factor*norms
		warped_points = [dist_center[0],dist_center[1]]+(nom/denom)*delta
		
		
	
	return warped_points