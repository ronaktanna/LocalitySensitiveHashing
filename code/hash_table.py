import numpy as np

class HashTable:
	def __init__(self, k_hash_size, feature_count, w_parameter = 0.078):
		"""
		Method Explanation:
			. Initializes properties for the current instance.
		Input(s):
			k_hash_size -- number of hash functions in the layer.
			feature_count -- number of features in the dataset.
			w_parameter -- the width of each bucket in the layer.
		"""
		self.k_hash_size = k_hash_size # number of hash functions in the layer.
		self.feature_count = feature_count # number of features, used for generating random projections 
		self.hash_table = dict() # The dictionary representing the data of the layer
		self.projections = self.init_projections() # The matrix of 'k' randomly generated projections
		self.w_parameter = w_parameter # The width of each bucket in the layer.
		self.b_offsets = self.init_b_offsets() # 'k' number of random shifts.
		print('W: ', w_parameter, ' self.w: ', self.w_parameter)
	def init_projections(self):
		"""
		Method Explanation:
			. Initializes the projections matrix comprising of 'k' random vectors of unit length.
			. The values in all random unit vectors are sampled from a normal distribution.
		"""
		the_projections = np.random.randn(self.k_hash_size, self.feature_count)
		for index, row_data in enumerate(the_projections):
			the_norm = np.linalg.norm(row_data)
			the_projections[index] = np.true_divide(row_data, the_norm)
		
		return the_projections

	def init_b_offsets(self):
		"""
		Method Explanation:
			. Initializes 'k' number of b_offsets sampled uniformly between 0 and w_parameter of the instance.
		"""
		to_return = list()
		for index in range(self.k_hash_size):
			to_return.append(np.random.uniform(0, self.w_parameter))
		return to_return

	def generate_hash(self, input_vector):
		"""
		Method Explanation:
			. Generate a hash value based on the euclidean hash family formula.
			. Each hash function generates a hash value of 11 bits long.
			. For k hash functions, we get a hash code of 11*k bits.
			. Used for filling the hash tables, querying them and also for nearest neighbor search.
		Input(s):
			input_vector -- The image represented as a vector.
		Output:
			The bit representation of the hash code that is generated cast to an integer representation comprising of 0s and 1s.
		"""
		hash_code = ""
		for index, row_data in enumerate(self.projections):
			random_vector_transpose = row_data.transpose()
			current_hash = np.floor((np.dot(input_vector, random_vector_transpose) + self.b_offsets[index])/self.w_parameter).astype("int")
			bit_representation = np.binary_repr(current_hash, 11)
			hash_code+= bit_representation
		return hash_code

	def __getitem__(self, input_vector):
		"""
		Method Explanation:
			. Gets the list of images in the same bucket as the query image in the layer.
		Input(s):
			input_vector -- the query image represented as a vector.
		"""
		hash_value = self.generate_hash(input_vector)
		return self.hash_table.get(hash_value, [])

	def __setitem__(self, input_vector, label):
		"""
		Method Explanation:
			. Generates the hash of the query image and appends it to the list of images falling in the same hash bucket.
		"""
		hash_value = self.generate_hash(input_vector) # Generate a hash value based on random projection 
		self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label] # Get all the items from the bucket as a list and append the label to that list.

	def get_reduced_hash_code(self, current_hash_code, k_value):
		"""
		Method Explanation:
			. Assists in nearest neighbor search.
			. Used by LSH class' get_t_candidates method.
			. Gives a reduced representation of the hash with 11, 22, 33... bits subtracted from the end depending on the k_value
		Input(s):
			current_hash_code -- the current hash_code of size 11*self.k_hash_size
			k_value -- a value lesser than self.k_hash_size
		Output:
			A reduced representation of the current_hash_code.
		"""
		if (k_value == self.k_hash_size) or (k_value == 0) or (self.k_hash_size - k_value < 0):
			return current_hash_code
		return current_hash_code[:(len(current_hash_code)-11*(self.k_hash_size - k_value))]
	
	def get_item_for_reduced_k(self, input_vector, k_value):
		"""
		DEPRECATED! KEPT IT ANYWAY AS A BACKUP.
		REFER TO LSH CLASS' get_t_candidates METHOD FOR THE NEAREST NEIGHBOR SEARCH IMPLEMENTATION.
		Method Explanation:
			. A helper for nearest neighbor search where if enough candidates are not retrieved for the current k_value, we reduce k iteratively
			  to get all candidate nearest neighbors.
			. This works only for a given k_value. The procedure to iteratively reduce k until al
		Input(s):
			input_vector -- The query image represented as a vector.
			k_value -- A value of the number of hash functions that is lesser than self.k_hash_size
		"""
		result_list = list()
		hash_code_key = self.generate_hash(input_vector)
		reduced_hash_code_key = self.get_reduced_hash_code(hash_code_key, k_value)
		
		for hash_code, imageid_list in self.hash_table.items():
			reduced_hash_code = self.get_reduced_hash_code(hash_code, k_value)
			if reduced_hash_code_key == reduced_hash_code:
				result_list.extend(imageid_list)
		
		return result_list