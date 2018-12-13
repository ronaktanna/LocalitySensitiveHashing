"""
Data parsing methods.
"""
from collections import OrderedDict
import constants
import xml.etree.ElementTree as et

class DataExtractor(object): 
	def location_mapping(self):
		tree = et.parse(constants.DEVSET_TOPICS_DIR_PATH)
		doc = tree.getroot()
		mapping = OrderedDict({})
		for topic in doc:
			mapping[topic.find("number").text] = topic.find("title").text

		return mapping