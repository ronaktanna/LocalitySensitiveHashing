from hash_table import HashTable
from LSH import LSH
import constants

class Driver:
	def __init__(self):
		self.L = int()
		self.k = int()
		self.lsh = None
		self.query_imageid = ''
		self.t = int()

	def runner(self):
		self.L = int(input('Enter the number of layers (L): '))
		self.k = int(input('Enter the number of Hashes per layer (k): '))
		self.lsh = LSH(self.L, self.k)

		for table_instance in self.lsh.hash_tables:
			print('Number of hash codes/buckets for the given layer: ', len(list(table_instance.hash_table.keys()))) #, ' Max size of any given bucket: ', max(list(table_instance.hash_table.values())))
			print('------------')
		print('')

		t_nearest_neighbors = list()
		returned_dict = dict()
		
		while True:
			self.query_imageid = int(input('Enter the image ID: '))
			self.t = int(input('Enter the number of nearest neighbors desired (t): '))
			returned_dict = self.lsh.get_atleast_t_candidate_nearest_neighbors(self.query_imageid, self.t)
			print('Total images considered: ', returned_dict['total_images_considered'])
			print('Unique images considered: ', returned_dict['unique_images_considered'])

			nearest_neighbors_list = self.lsh.get_t_nearest_neighbors(self.query_imageid, returned_dict['result_list'], self.t)
			for nearest_neighbor in nearest_neighbors_list: # Get the image IDs alone
				t_nearest_neighbors.append(nearest_neighbor['image_id'])
			
			print('The T nearest neighbors: ', t_nearest_neighbors, '\n')
			runagain = input('Run again? (Y/N): ')
			t_nearest_neighbors.clear()
			returned_dict.clear()
			if runagain == 'N':
				break

if __name__ == "__main__":
	driver = Driver()
	driver.runner()