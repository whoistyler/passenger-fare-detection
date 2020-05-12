import numpy as np

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.centroid_mean = 0
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
	
	def update_mean(self):
		self.centroid_mean = (np.mean(self.centroids, axis=0))[1]