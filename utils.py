## Author: Phạm Minh Hiếu
## Functions: Xử lý những tác vụ liên quan đển file dữ liệu như đọc file, chuyển đổi nhãn - File handling
from scipy.io import arff
import pandas as pd


def read_arff(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    return df

def convert_label(label_str):
    if type(label_str).__name__ == "bytes":
        label_str = label_str.decode()
    
    if label_str.lower() == 'true':
        return 1
    else:
        return 0

if __name__ == "__main__":
    path = "data/thesis_data/arff/pc1.arff.txt"
    data = read_arff(path)
    print(type(data.defects[0]).__name__)