import pandas as pd
import numpy as np
# import ReadRequest as RQ
import ContorTedad
import os
df = pd.read_csv("Trusted1.csv")
"/Users/amirsajjad/Desktop/CarID_OCR/fashtkhudam/"
koli = 0
doros = 0
ghalat = 0
bishtarpeidakardim = 0
cheghadkamtar = 0
kolma = 0
kolann = 0
for pic in os.listdir("/Users/amirsajjad/Desktop/CarID_OCR/fasht"):
    try:
        print(pic)
        inja = df.loc[df["Name"] == pic]["Tedad"].values[0]
        print("/Users/amirsajjad/Desktop/CarID_OCR/fasht/"+pic)
        ContorTedad.ReadReq(["/Users/amirsajjad/Desktop/CarID_OCR/fasht/"+pic])
        print("Finder",inja,ContorTedad.Platetimes,pic)
        koli+=1
        kolma += ContorTedad.Platetimes
        kolann +=inja
        if inja <= ContorTedad.Platetimes:
            print("miad inja")
            doros+=1
            bishtarpeidakardim += min(ContorTedad.Platetimes,5) - inja
        else:
            ghalat+=1
            cheghadkamtar += inja - ContorTedad.Platetimes
        # ContorTedad.ReadReq([pic])
    except:
        pass
    print("Natije", koli, doros, ghalat)
    print("Natije",bishtarpeidakardim,cheghadkamtar)

print("Natije",koli,doros,ghalat)
print("Natije",bishtarpeidakardim,cheghadkamtar)
print(kolann,kolma)