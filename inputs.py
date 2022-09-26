import numpy as np

def is_clean(line: str) -> bool:
    for data_entry in line.split(","):
        if data_entry == "?":
            return False
    
    return True


# Import hepatitis data
__hepatitis_data = open("./data/hepatitis.data")

hepatitis_clean_data: list[str] = []

for __line in __hepatitis_data.readlines():
    __line = __line.strip('\n')
    if(is_clean(__line)):
        __new_line: list[float] = []
        for __data_point in __line.split(','):
            __new_line.append(float(__data_point))
        hepatitis_clean_data.append(__new_line)

hepatitis_clean_data = np.array(hepatitis_clean_data)

__hepatitis_data.close()



# Import diabetes data
__diabetes_data = open("./data/messidor_features.arff")

diabetes_clean_data: list[str] = []

__read: bool = False
for __line in __diabetes_data.readlines():
    __line = __line.strip('\n')
    if __read and is_clean(__line):
        __new_line: list[float] = []
        for __data_point in __line.split(','):
            __new_line.append(float(__data_point))
        diabetes_clean_data.append(__new_line)

    else:
        __read = __line == "@data"

diabetes_clean_data = np.array(diabetes_clean_data)

__diabetes_data.close()