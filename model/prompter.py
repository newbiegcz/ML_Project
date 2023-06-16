import torch
import cv2 as cv
import numpy as np
# model/prompter.py
# 生成 prompt
# 可能是一个可训练的模块
class Prompter(torch.nn.Module):
	def forward(self, mask, num = 1):
		'''mask 是当前器官的一个 mask
			 0 表示该像素不在当前器官内，1 表示在器官内
			 返回一个可以直接给 sam.predict 的 dict
			   表示选择的 prompt'''
		result = dict()
		result["multimask_output"] = True
		if num == 1:
			dst = cv.distanceTransform(mask, cv.DIST_L2, 3)
			ind = np.unravel_index(np.argmax(dst, axis = None), dst.shape)
			print(ind[0], ind[1])
			result["point_coords"] = np.array([ind])
			result["point_labels"] = np.array([1])
		else:
			x, y = mask.shape
			coords = np.zeros((num, 2))
			labels = np.zeros(num)
			for i in range(num):
				if i % 10 == 0:
					while(True):
						ax, ay = np.random.rand(2)
						cx = int(ax * x)
						cy = int(ay * y)
						if mask[cx][cy] == 1:
							break
					coords[i][0] = cx
					coords[i][1] = cy
					labels[i] = 1
				else:
					ax, ay = np.random.rand(2)
					cx = int(ax * x)
					cy = int(ay * y)
					coords[i][0] = cx
					coords[i][1] = cy
					if mask[cx][cy] == 1:
						labels[i] = 1
					else:
						labels[i] = -1

			result["point_coords"] = coords
			result["point_labels"] = labels

		return result
		