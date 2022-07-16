import json
import os
import subprocess
import ConFindercopy
import cv2
import Partioningcopy
import numpy as np
import tensorflow as tf
import time
import index
new_model = tf.keras.models.load_model('pNCompleteDatasetModel.h5')



def PorNP(Contur):
    # print(Contur)
    print("PLate or not plate Started")
    res = []
    shahid = []
    for i in Contur:
        # print("saeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeedsaeeed",i.shape[1]/i.shape[0])
        input_array = cv2.resize(i,(32,32))
        input_array = np.reshape(input_array, (32, 32,1))
        shahid.append(input_array)
    # input_array = np.expand_dims(input_array, axis=0)
        # print("havij",input_array.shape)
        # print(new_model.predict(input_array)[0][0])
    # print(len(shahid))
    shash = []
    sad = 0
    # print(new_model.predict(np.reshape(shahid, (len(shahid),32, 32,1))))
    for am in new_model(np.reshape(shahid, (len(shahid),32, 32,1))):
        if am < 0.3 and i.shape[1] / i.shape[0] < 3.9:
            res.append(1)
        else:
            if am[0] < 0.3:
                sa = time.time()
                lists = Contur[sad].tolist()
                # json_str = json.dumps(lists)
                json_str = str(lists)
                shash.append(json_str)
                print("kale ghandi",time.time() - sa)
            res.append(am[0])
        sad += 1
        # age w/h bod bozorg taresh o bar midarim
    print("PLate or not plate Started")
    index.index("TrustedCON",shash)

    # return shash
