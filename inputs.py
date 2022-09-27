import numpy as np
import os

def is_clean(line: str) -> bool:
    for data_entry in line.split(","):
        if data_entry == "?":
            return False
    
    return True


# Import hepatitis data
__hepatitis_data = open(os.path.dirname(os.path.abspath(__file__)) + "/data/hepatitis.data")

hepatitis_clean_data: list[str] = []

for __line in __hepatitis_data.readlines():
    __line = __line.strip('\n')
    if(is_clean(__line)):
        hepatitis_clean_data.append(__line.split(','))

hepatitis_clean_data = np.array(hepatitis_clean_data, float)

__hepatitis_data.close()



# Import diabetes data
__diabetes_data = open(os.path.dirname(os.path.abspath(__file__)) + "/data/messidor_features.arff")

diabetes_clean_data: list[str] = []

__read: bool = False
for __line in __diabetes_data.readlines():
    __line = __line.strip('\n')
    if __read and is_clean(__line):
        diabetes_clean_data.append(__line.split(','))

    else:
        __read = __line == "@data"

diabetes_clean_data = np.array(diabetes_clean_data, float)

__diabetes_data.close()