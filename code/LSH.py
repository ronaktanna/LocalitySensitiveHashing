import numpy as np
import pandas as pd
from operator import itemgetter
from hash_table import HashTable
from preprocessor import PreProcessor

""" Euclidean Hash Family => H(v) = |_(r.v + b)/w_| --- (1) """
class LSH:
	def __init__(self, L_layer_count, k_hash_functions_per_layer, w_parameter = 0.20):
		self.preprocessor = PreProcessor() # initialize the preprocessor instance
		self.data_dict = dict() # declaring a dicitonary where key is image id and value is list of features. The values will be indexed.
		self.L_layer_count = L_layer_count # the number of layers
		self.k_hash_functions_per_layer = k_hash_functions_per_layer # number of hash functions per layer
		self.feature_count = 0 # feature_count, will be given a value in init_data(). Used to generate the random projections.
		self.image_ids = list() # The list of image IDs from the data_dict maintained separately. Ordering isn't a problem as in Python 3.7, dictionaries are ordered in terms of insertion.
		self.data_matrix = [] # The data matrix of features alone
		self.init_data() # Initializes the image IDs, data matrix and also the feature count as it is dependent on input data
		self.w_parameter = w_parameter # w in (1)
		self.hash_tables = list() # The list of hash tables or layers

		print("Initializing the hash tables...")
		for value in range(self.L_layer_count): # create L hash tables with k hash functions per layer
			self.hash_tables.append(HashTable(self.k_hash_functions_per_layer, self.feature_count, self.w_parameter))

		self.fill_all_hashtables() # Index all the data points in all the layers

	def init_data(self):
		"""
		Method Explanation:
			. Triggers the start of preprocessing, initializes the data_dict, image_ids, data_matrix and feature_count.
		"""
		data_dict = self.preprocessor.preprocess_MinMaxScaler()
		self.image_ids = list(data_dict.keys())
		self.data_matrix = np.asarray(list(data_dict.values()))
		print("\nNumber of rows: ", len(self.data_matrix), " Number of columns: ", len(self.data_matrix[0]), "\n")
		self.data_matrix = self.preprocessor.compute_latent_semantics(self.data_matrix)
		print("\nNumber of rows: ", len(self.data_matrix), " Number of columns: ", len(self.data_matrix[0]), "\n")

		for index, image_id in enumerate(self.image_ids):
			self.data_dict[image_id] = self.data_matrix[index]
		self.feature_count = len(self.data_matrix[0]) # features_retained

	def get_t_candidates_helper(self, input_vector, k_value):
		"""
		Method Explanation:
			. Returns the list of images in the same bucket as the given image accross all hash tables (layers) for
			a specific "simulated" value of k that is lesser than what the user inputted for k.
			. Helper for get_atleast_t_candidate_nearest_neighbors that runs the whole iterative process, whereas this method runs it only for
			a specific value of k.
		Input vector(s):
			input_vector -- the query image represented as a vector.
			k_value -- the value of number of hashes per layer that is lesser than what the user inputted.
		"""
		result_list = list()
		return_dict = dict()
		for table in self.hash_tables:
			hash_code_key = table.generate_hash(input_vector)
			reduced_hash_code_key = table.get_reduced_hash_code(hash_code_key, k_value)
			
			for hash_code, imageid_list in table.hash_table.items():
				reduced_hash_code = table.get_reduced_hash_code(hash_code, k_value)
				if reduced_hash_code_key == reduced_hash_code:
					result_list.extend(imageid_list)

		total_images_considered = len(result_list)
		result_list = list(set(result_list))
		unique_images_considered = len(result_list)
		return_dict = { "total_images_considered": total_images_considered, "unique_images_considered": unique_images_considered, "result_list": result_list }

		return return_dict
	
	def get_atleast_t_candidate_nearest_neighbors(self, image_id, t):
		"""
		Method Explanation:
			. Used for getting atleast t candidate nearest neighbors.
			. Starts with the user input of 'k' and tries to get 't' candidates as nearest neighbors.
			. If 't' nearest neighbor candidates aren't found, the method iteratively reduces the value of k (simulates a reduced k for querying)
			until atleast 't' of them are retrieved.
		Input(s):
			image_id -- the query image ID.
			t -- an integer representing the number of nearest neighbor candidates desired.
		"""
		candidate_list = list()
		returned_dict = dict()
		current_k = self.k_hash_functions_per_layer
		input_vector = self.data_dict[image_id] # representation of the image as a vector

		returned_dict = self.__getitem__(input_vector) # First try getting atleast 't' candidates for k = self.k_hash_functions_per_layer
		candidate_list = returned_dict["result_list"]
		if len(candidate_list) >= t:
			return returned_dict # we have atleast t candidate neighbors, return them
		
		print("Did not get enough candidates (", len(candidate_list), "), reducing k...")
		current_k-= 1 # reduce k and try again
		
		returned_dict.clear() # clear the dict for the new return values
		candidate_list.clear()
		while True:
			if current_k == 0:
				return self.image_ids # return all the images as candidates in the worst case
			returned_dict = self.get_t_candidates_helper(input_vector, current_k)
			candidate_list = returned_dict["result_list"]
			if len(candidate_list) >= t:
				return returned_dict # we have atleast t candidate neighbors, return them
			print("Did not get enough candidates (", len(candidate_list), "), reducing k...")
			current_k-= 1 # decrease k and try again
			returned_dict.clear()
			candidate_list.clear()

	def get_t_nearest_neighbors(self, query_imageid, candidate_imageids, t):
		"""
		Method Explanation:
			. Gets 't' nearest neighbors from the candidate imageids in the reduced search space.
			. Executed after getting atleast 't' candidates via get_atleast_t_candidate_nearest_neighbors()
		Input(s):
			query_imageid -- integer representing the image id of the query image.
			candidate_imageid -- list of integers representing the candidate image ids for nearest neighbor search.
			t -- integer representing the number of closest nearest neighbors desired.
		Output:
			A list of 't' nearest neighbor image ids.
		"""
		distance_list = list()
		query_image_vector = self.data_dict[query_imageid]
		for candidate_imageid in candidate_imageids:
			candidate_image_vector = self.data_dict[candidate_imageid] 
			distance_list.append({ "image_id": candidate_imageid, "distance": np.linalg.norm(query_image_vector - candidate_image_vector)})

		sorted_list = sorted(distance_list, key=itemgetter("distance"))
		return sorted_list[:t]


	def fill_the_hashtable(self, table_instance):
		"""
		Method Explanation:
			. Helper method for fill_all_hashtables.
			. Takes care of indexing the data for one layer.
			. Generates hashes for each data point and places them into its corresponding bucket.
		Input(s):
			table_instance -- The object representing a single layer.
		"""
		for index, image in enumerate(self.data_matrix):
			the_label = self.image_ids[index]
			table_instance.__setitem__(image, the_label)

	def fill_all_hashtables(self):
		"""
		Method Explanation:
			. Wrapper method over fill_the_hashtable helper.
			. Takes care of indexing the data for all layers.
			. Repeats L times: generates hashes for each data point and places them into its corresponding bucket.
		"""
		print("Filling the hash tables...")
		for table in self.hash_tables:
			self.fill_the_hashtable(table)
		print("Index structure ready...\n")

	def __getitem__(self, input_vector):
		"""
		Method Explanation:
			. Returns the list of images in the same bucket as the given image accross all hash tables (layers) for user's inputted k.
			. Used as a helper in get_atleast_t_candidate_nearest_neighbors as the first step of querying.
		Input(s):
			input_vector -- The representation of the image in the form of a vector.
		"""
		result_list = list()
		return_dict = dict()
		for table in self.hash_tables:
			result_list.extend(table[input_vector])
		
		total_images_considered = len(result_list)
		result_list = list(set(result_list))
		unique_images_considered = len(result_list)
		return_dict = { "total_images_considered": total_images_considered, "unique_images_considered": unique_images_considered, "result_list": result_list }

		return return_dict # list(set(result_list))

	def __setitem__(self, input_vector, label):
		"""
		Method Explanation:
			. Custom implementation of the setter that sets the value of the image represented
			by the input_vector accross all hash tables.
		Input(s):
			input_vector -- The representation of the image in the form of a vector.
			label -- the label you want to give the image when it is placed in the bucket. In this case, the ImageID.
		"""
		for table in self.hash_tables:
			table[input_vector] = label