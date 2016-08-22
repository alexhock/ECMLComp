__author__ = 'alexh'
import numpy as np
#import xgboost as xgb

root_folder = 'c:/users/alexh/projects/xgb/NetCla/NetCla/data/'

# label need to be 0 to num_class -1
#data = np.loadtxt(root_folder + '/dermatology.data', delimiter=',',converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1 } )
#data = np.loadtxt(root_folder + '/train.csv', delimiter='\t', skiprows=1)
data = np.loadtxt(root_folder + '/train_target.csv')
data2 = np.loadtxt(root_folder + '/valid_target.csv')

print ("{0} {1}".format(max(data), max(data2)))

sz = data.shape

#train = data[:int(sz[0] * 0.7), :]
#test = data[int(sz[0] * 0.7):, :]

#train_X = train[:,0:33]
#train_Y = train[:, 34]


test_X = test[:,0:33]
test_Y = test[:, 34]
