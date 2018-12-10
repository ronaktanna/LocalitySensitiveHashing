"""
ALGORITHM FOR PREPROCESSING:
---------------------------

. We first concatenate all the features from the following visual models in the dataset for
  every unique image: CM3x3, CN, CSD, LBP3x3, HOG, GLRLM.
	. The choice is because of the simple intuition that we combine 3 models describing the aspect of color
	  in the image (CM3x3, CN, CSD) and 3 models describing the aspect of texture/shape (LBP3x3, HOG, GLRLM)
	  in the image.
. Now we will have 425 features for every image.
. We compute latent semantics of this data using the Singular Valued Decomposition (SVD) and retain all the concepts
  with an eigen value >=1. We now have 364 latent semantics of the data that represent a combination of shape/texture
  and color. 
	. The choice of removing all the semantics whose eigen values < 1 is because the semantic concepts with eigen values < 1
	  might correspond to noisy data.

"""

from data_extractor import DataExtractor
import constants
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'

class PreProcessor:

	def __init__(self):
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()
		self.location_names = list(self.mapping.values())
		self.reference_model = 'CM3x3'
		self.model_list = self.init_model_list()
		self.reference_df = pd.DataFrame()
		self.df_list = self.init_df_list()
		self.data_dict = dict()
		self.minmax_scaler = MinMaxScaler()

	def init_model_list(self):
		"""
		Method Explanation:
			. Initializes the model_list as every model name other than the reference model for preprocessing.
		"""		
		models = ['CN', 'CSD', 'LBP3x3', 'HOG', 'GLRLM'] # along with reference model CM3x3
		
		return models
	
	def init_df_list(self):
		"""
		Method Explanation:
			. Initializes the df_list comprising of the dataframes.
		"""
		to_return = list()
		for model in self.model_list:
			to_return.append(pd.DataFrame())
		return to_return

	def compute_first_index_lesser_than_one(self, S):
		"""
		Method Explanation:
			. Computes the first index in S that is lesser than 1 where S is a vector representation of the
			factor matrix in SVD with eigen values in decreasing order of weights.
		Input(s):
			S -- Vector representation of the factor matrix in SVD (S in U, S, Vt) comprising of eigen values
				sorted in decreasing order of weights.
		Output(s):
			The number of eigen values to consider for concept mapping given by the first index in S with a
			value lesser than 1.
		"""
		for index, eigen_value in enumerate(S):
			if eigen_value < 1:
				return index+1

	def compute_latent_semantics(self, feature_matrix):
		"""
		Method Explanation:
			. Returns the latent semantic representation of the feature_matrix with 'k' concepts.
			. 'k' -- number of concepts -- index of the first eigen value that is lesser than
			   1 in S represented as a vector in decreasing order of weights.
		Input(s):
			feature_matrix -- the list of all features of all image IDs on top of which SVD would be done.
		Output:
			The concept mapping of the feature_matrix in 'k' dimensions/concepts.
		"""

		print('Finding latent semantics of the data...')
		U, S, Vt = np.linalg.svd(feature_matrix)
		
		print('Removing eigen vectors with eigen value less than 1...')
		k = self.compute_first_index_lesser_than_one(S)
		S = np.diag(S)

		print('Preprocessing done...')
		return (np.dot(U[:,:k], S[:k,:k]))
		
	def preprocess_MinMaxScaler(self):
		"""
		Method Explanation:
			. Refer to the top of the file for the algorithm for preprocessing.
			. Uses the MinMaxScaling for the normalization of data between 0 and 1.
		"""
		print('\nPreprocessing the data...')
		self.data_dict.clear()
		# print('Current model being processed: ', self.reference_model, '...')
		
		for location in self.location_names:
			current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + self.reference_model + ".csv", header = None)
			self.reference_df = self.reference_df.append(current_df, ignore_index = True)

		self.reference_df = self.reference_df.drop_duplicates(subset=[0], keep = 'first') # drop duplicate image ID rows and keep the first one.
		columns_to_normalize = np.arange(1, self.reference_df.shape[1], 1) # the column indices to which MinMax normalization will be applied to.
		self.reference_df[columns_to_normalize] = self.minmax_scaler.fit_transform(self.reference_df[columns_to_normalize]) # MinMax normalization

		self.data_dict = self.reference_df.set_index(0).T.to_dict('list') # Filling the data dict

		temp_dict = dict()
		for index, model in enumerate(self.model_list):
			# print('Current model being processed: ', model, '...')
			for location in self.location_names:
				# print('\tLocation being processed: ', location, '...')
				current_df = pd.read_csv("../dataset/visual_descriptors/" + location + " " + model + ".csv", header = None)
				df_to_modify = self.df_list[index] # Get the current model's DF that has been populated with X items so far...
				df_to_modify = df_to_modify.append(current_df, ignore_index = True) # Append the current df to the current model's DF...
				self.df_list[index] = df_to_modify
			model_df = self.df_list[index]

			model_df = model_df.drop_duplicates(subset=[0], keep='first') # drop duplicate image ID rows and keep the first one.
			columns_to_normalize = np.arange(1, model_df.shape[1], 1) # the column indices to which MinMax normalization will be applied to.
			model_df[columns_to_normalize] = self.minmax_scaler.fit_transform(model_df[columns_to_normalize])
			self.df_list[index] = model_df

			temp_dict = self.df_list[index].set_index(0).T.to_dict('list')

			for key, val in temp_dict.items():
				if key in self.data_dict:
					current_list = self.data_dict[key]
					current_list.extend(temp_dict[key])
					self.data_dict[key] = current_list # insert into data dict only if image id is already present

			temp_dict.clear() # clear the temp dictionary for the next iteration
		
		# Apply MinMax scaling to the feature concatenated data matrix as well
		the_feature_matrix = self.minmax_scaler.fit_transform(np.asarray(list(self.data_dict.values())))

		# Repopulate the data_dict
		index_counter = 0
		for key, val in self.data_dict.items():
			self.data_dict[key].clear()
			self.data_dict[key] = the_feature_matrix[index_counter]
			index_counter+= 1

		return self.data_dict