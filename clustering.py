import numpy
import random as rand
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


def distance(p1, p2):
	if len(p1) != len(p2):
		print("Dimension of points do not match...")
		return 

	d, dim = 0.0, len(p1)
	for i in range(dim):
		d += (p1[i]-p2[i])**2

	return math.sqrt(d)


def mean(points):
	if not points:
		print("No points to compute on...")
		return

	# Initialize vars
	count      = len(points)
	dim 	   = len(points[0])
	mean_point = [0.0] * dim

	for d in range(dim):
		for p in points:
			mean_point[d] += p[d]
		mean_point[d] /= count

	return mean_point


def cost(centroid_info):
	J = 0.0
	for c in centroid_info:
		J += c[1]**2
	return J / len(centroid_info)


def is_change(old_points, new_points):
	if len(old_points) != len(new_points) or not old_points or not new_points:
		return True

	points_count = len(new_points)
	dim 		 = len(new_points[0])

	for p in range(points_count):
		for d in range(dim):
			if old_points[p][d] != new_points[p][d]:
				return True

	return False


def kmeans(data = [], k = 2):
	##################################
	###### INITIALIZE VARIABLES ######
	##################################
	data_size 			= len(data)  # size of dataset
	centroid_positions  = [] 		 # array of centroid positions
	is_centroid_change  = True		 # keep track of centroid positions change
	if data_size == 0:
		raise Exception("No data provided...")

	# 1. Randomly initialize k centroids (i.e. initial guesses for centroid positions)
	for _ in range(k):
		rand_i = rand.randrange(data_size)
		centroid_positions.append(data[rand_i])
	
	# loop while no more change in centroid positions
	while is_centroid_change:

		# 2. Compute distance b/w each datapoint and centroids -> dataIndex:[distance1, distance2, ...]
		distances_to_centroids = []
		for d in range(data_size):
			c_pos = []
			for c in centroid_positions:
				c_pos.append(distance(data[d], c))
			distances_to_centroids.append(c_pos)

		# 3a. Assign datapoint to centroid that is closest to it -> dataIndex:[centroidIndex, distance]
		for d in range(data_size):
			min_val = min(distances_to_centroids[d])
			distances_to_centroids[d] = [distances_to_centroids[d].index(min_val), min_val]

		# 3b. Assign data coordinates to centroids -> centroidIndex: [[data1], [data2], ...]
		cluster_data_points = {}
		for d in range(data_size):
			if distances_to_centroids[d][0] not in cluster_data_points:
				cluster_data_points[distances_to_centroids[d][0]] = [data[d]]
			else:
				cluster_data_points[distances_to_centroids[d][0]].append(data[d])
		
		# 4b. Update location of cluster centroids (i.e. move the cluster centroid to the mean position)
		new_positions = [[]] * (1 + max(cluster_data_points.keys()))
		for _k in cluster_data_points.keys():
			new_positions[_k] = mean(cluster_data_points[_k])
		
		if is_change(centroid_positions, new_positions):
			is_centroid_change = True
			centroid_positions = new_positions.copy()
		else:
			is_centroid_change = False

	return centroid_positions, cluster_data_points, cost(distances_to_centroids)


def kmeans_distortion_plot(x_label, x_data, y_label, y_data):
	plt.plot(x_data, y_data, 'ro')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()


if __name__ == "__main__":
	# load in data as numpy array of n-dim datapoints
	dataset = load_breast_cancer()['data']

	cluster_centroid_locations, cluster_centroid_assignments, j_cost = kmeans(data=dataset, k=2)

	iters, costs = list(range(2, 8)), []

	for i in iters:
		cluster_centroid_locations, cluster_centroid_assignments, j_cost = kmeans(data=dataset, k=i)
		costs.append(j_cost)

	kmeans_distortion_plot(x_label='k-Values', x_data=iters, y_label='Distortion (J-Cost)', y_data=costs)