import torch
import cv2 as cv
import numpy as np
# model/prompter.py
# 生成 prompt
# 可能是一个可训练的模块
def bounding_box(mask):
	c = np.nonzero(mask)
	x1 = np.min(c[0])
	x2 = np.max(c[0])
	y1 = np.min(c[1])
	y2 = np.max(c[1])
	return x1, x2, y1, y2

class Prompter(torch.nn.Module):
	def forward(self, mask, sam_predictor, num, points = True, box = False):
		'''
		Args:
			mask: segmentation mask
			sam_predictor: sam_predictor
			num: number of prompt points
			points: whether to use points as prompt
			box: whether to use box as prompt

		Returns:
			result: prompt result

		'''
		result = dict()
		result["multimask_output"] = False
		if points == True:
			coords = np.zeros((num, 2))
			labels = np.zeros(num)
			dst = cv.distanceTransform(mask, cv.DIST_L2, 3)
			ind = np.unravel_index(np.argmax(dst, axis = None), dst.shape)
			coords[0][0] = ind[1]
			coords[0][1] = ind[0]
			labels[0] = 1
			for i in range(1, num):
				result["point_coords"] = coords
				result["point_labels"] = labels
				msk, score, logit = sam_predictor.predict(**result)
				x, y, z = msk.shape
				msk = msk.reshape(y, z)
				dst = cv.distanceTransform(mask ^ msk, cv.DIST_L2, 3)
				ind = np.unravel_index(np.argmax(dst, axis = None), dst.shape)
				coords[i][0] = ind[1]
				coords[i][1] = ind[0]
				if mask[ind] == 1:
					labels[i] = 1
				else:
					labels[i] = 0

			result["point_coords"] = coords
			result["point_labels"] = labels
		else:
			assert(box == True)
			x1, x2, y1, y2 = bounding_box(mask)
			result["box"] = np.array([y1 - np.random.randint(0, 10), x1 - np.random.randint(0, 10), y2 + np.random.randint(0, 10), x2 + np.random.randint(0, 10)])

		return result
		