import pickle
def init():
    for i in ["HaveRead","Haveconfinded","HaveConPlate","HaveNumber"]:
        filename = i+'.pk'
        with open(filename, 'wb') as fi:
            pickle.dump([], fi)

# init()
def init1():

    for i in ["HaveRead","Haveconfinded","HaveConPlate","HaveNumber"]:
        # try:
        filename = i+'.pk'
        with open(filename, 'rb') as fi:
            HaveRead = pickle.load(fi)
            print(i)
            print(len(HaveRead))
            fi.close()
        # except:
        #     pass

init1()

# while True:
#     init1()

    # global HaveRead,Haveconfinded,HaveConPlate,HaveNumber
    # HaveRead = []
    # Haveconfinded = []
    # HaveConPlate = []
    # HaveNumber = []
