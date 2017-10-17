from random import seed
from random import randrange
from csv import reader
import pandas as pd
import numpy as np


def get_gini(groups, classes):
	""" return weighted gini index
	"""
	n_instances = np.sum([len(group) for group in groups])
	gini = 0.0
	for group in groups:
		if len(group) == 0:
			continue
		instances = [inst[-1] for inst in group]
		score = 0.0
		for class_val in classes:
			prop = 1.0 * instances.count(class_val) / len(instances)
			score +=  prop * prop
		gini += (1.0 - score) * len(instances) / n_instances
	return gini

def split(dataset, col_index, value):
	""" perform node split
	"""
	left, right = [], []
	for row in dataset:
		if row[col_index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def split_parent(dataset):
	""" binary split
	"""
	classes = list(set([row[-1] for row in dataset]))
	min_gini, index, value,  groups = 999, None, None, None
	for col_index in range(len(dataset[0]) - 1):
		for row in dataset:
			left, right = split(dataset, col_index, row[col_index])
			gini = get_gini([left, right], classes)
			if gini < min_gini:
				min_gini = gini
				index, value = col_index, row[col_index]
				groups = [left, right]
	return {"col_index" : index, "value" : value,
			"groups" : groups}

def to_terminal(group):
	""" return the value of a leaf node
	"""
	res = [inst[-1] for inst in group]
	return max(set(res), key=res.count)

def split_children(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check if no node split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# if it is max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = split_parent(left)
		split_children(node['left'], max_depth, min_size, depth+1)
	# process right
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = split_parent(right)
		split_children(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
	root = split_parent(train)
	split_children(root, max_depth, min_size, 1)
	return root

def print_tree(node, depth=1):
	if isinstance(node, dict):
		print ("{}index = {}, value = {}".format(depth*"   ", node['col_index'], node['value']))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print("{}res = {}".format(depth*"   ", node))

def predict(node, row):
	if isinstance(node, dict):
		if row[node['col_index']] < node['value']:
			return predict(node['left'], row)
		else:
			return predict(node['right'], row)
	else:
		return node

def decision_tree(train, test, max_depth, min_size):
	root = build_tree(train, max_depth, min_size)
	res = list()
	for row in test:
		res.append(predict(root, row))
	return res	

# test Gini values
print(get_gini([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(get_gini([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

# build a tree model
dataset = [[2.771244718,1.784783929,0],
			[1.728571309,1.169761413,0],
			[3.678319846,2.81281357,0],
			[3.961043357,2.61995032,0],
			[2.999208922,2.209014212,0],
			[7.497545867,3.162953546,1],
			[9.00220326,3.339047188,1],
			[7.444542326,0.476683375,1],
			[10.12493903,3.234550982,1],
			[6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)

tree = build_tree(dataset, 2, 1)
print_tree(tree)

#  predict with a stump
stump = {'col_index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for row in dataset:
	prediction = predict(stump, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))




# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	""" k-fold split
	"""
	fold_size = int(len(dataset) / n_folds)
	dataset_copy = list(dataset)
	dataset_split = list()
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def get_accuracy(actual, predicted):
	count = np.sum(np.array(actual) == np.array(predicted))
	return 1.0 * count / len(actual)

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	dataset = [list(row) for row in dataset.values]
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train = list(folds)
		train.remove(fold)
		train = sum(train, [])
		test = list(fold)
		predicted = algorithm(train, test, *args)
		actual = [row[-1] for row in test]
		scores.append(get_accuracy(actual, predicted))
	return scores

df = pd.read_csv("data_banknote_authentication.txt", header=None)
print (df.info())
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(df, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

