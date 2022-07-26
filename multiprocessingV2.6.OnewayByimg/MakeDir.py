import os
import cv2


while



def makeDIR(parent_dir,path,lastres,contures,img):
    directory = path.split("/")[-1]
    path1 = os.path.join(parent_dir, directory)
    os.mkdir(path1)
    cv2.imwrite(path1 + "/"+"directory.png", img)
    count = 0
    count1 = 0
    dir = []
    path3 = os.path.join(path1, str(0))
    os.mkdir(path3)
    for i,j in zip(lastres,contures):
        # print(type(i),int(i))
        if float(i) < 0.3:

            count += 1
            path2 = os.path.join(path1, str(count))
            dir.append(path2)
            os.mkdir(path2)
            cv2.imwrite(path2+"/"+"Pelak"+str(count)+".png",j)
        else:
            count1 += 1
            dir.append(path3)
            cv2.imwrite(path3 +"/"+"Pelak"+ str(count1)+".png", j)
    return dir