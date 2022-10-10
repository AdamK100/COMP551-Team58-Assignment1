import numpy as np
from helpers import entropy, evaluate_acc, manhattan_distance, misclass_rate, remove_irrelevant_features
import models
import inputs
from plots import plot


#Computing Statistics on Datasets

#DataSet 1
distrib1 = np.bincount((inputs.hepatitis_clean_data[:,0]).astype(int))
print("\nDataset 1 Statistics:")
print('Class 1 (Die) data ratio: ' + str(distrib1[1]/(float)(len(inputs.hepatitis_clean_data[:,0]))))
print('Class 2 (Live) data ratio: ' + str(distrib1[2]/(float)(len(inputs.hepatitis_clean_data[:,0]))))
print('Attrib 1 (Age) Mean: ' + str(np.mean(inputs.hepatitis_clean_data[:,1])))
sdistrib = np.bincount((inputs.hepatitis_clean_data[:,2]).astype(int))
print('Attrib 2 (Sex) distribution: Male = ' + str(sdistrib[1]/(float)(len(inputs.hepatitis_clean_data[:,2]))) + ", Female = " + str(sdistrib[2]/(float)(len(inputs.hepatitis_clean_data[:,2]))))

print("\nDataset 2 Statistics:")
distrib2 = np.bincount((inputs.diabetes_clean_data[:,-1]).astype(int))
print('Class 0 data ratio: ' + str(distrib2[0]/(float)(len(inputs.diabetes_clean_data[:,-1]))))
print('Class 1 data ratio: ' + str(distrib2[1]/(float)(len(inputs.diabetes_clean_data[:,-1]))))
adistrib = np.bincount((inputs.diabetes_clean_data[:,0]).astype(int))
print('Attrib 0 distribution: \'0\' = ' + str(adistrib[0]/(float)(len(inputs.diabetes_clean_data[:,0]))) + ", \'1\' = " + str(adistrib[1]/(float)(len(inputs.diabetes_clean_data[:,0]))))
adistrib2 = np.bincount((inputs.diabetes_clean_data[:,1]).astype(int))
print('Attrib 1 distribution: \'0\' = ' + str(adistrib2[0]/(float)(len(inputs.diabetes_clean_data[:,1]))) + ", \'1\' = " + str(adistrib2[1]/(float)(len(inputs.diabetes_clean_data[:,1]))))
print('Attrib 2 Mean: ' + str(np.mean(inputs.diabetes_clean_data[:,2])))

#Test 1: Accuracy of Models ()
