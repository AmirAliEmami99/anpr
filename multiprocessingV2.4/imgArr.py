import cv2
import json
def ReadImage1(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


sag = ReadImage1("Tessting/imag21.jpg")

# lists = sag.tolist()
json_str = json.dumps(sag)