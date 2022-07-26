import tkinter as tk
import tkinter.filedialog as fd
import ReadRequest
# import GUI
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

maindic = {}
con = 1
li = []
neshangar = 0
def changeimage():
    global neshangar
    labelList = [0] * len(maindic[li[neshangar]]['number'])
    labelListPlate = [0] * len(maindic[li[neshangar]]['number'])
    imageList = [0] * len(maindic[li[neshangar]]['number'])
    shamorti = [0] * len(maindic[li[neshangar]]['number'])
    contor = 0
    kas = Image.fromarray(maindic[li[neshangar]]['drawedimg'])
    kas = kas.resize((800, 800), Image.ANTIALIAS)
    kas = ImageTk.PhotoImage(kas)
    my_lbl.config(image=kas)
    my_lbl.image = kas
    for i in range(0, len(maindic[li[neshangar]]['number'])):
        imageList[i] = Image.fromarray(maindic[li[neshangar]]['pelakha'][i])
        imageList[i] = imageList[i].resize((200, 60), Image.ANTIALIAS)
        shamorti[i] = ImageTk.PhotoImage(imageList[i])
        labellistt[i * 2].configure(image=shamorti[i])
        labellistt[i * 2].image = shamorti[i]
        # labellistt[i].grid(row=i * 2, column=contor)
        contor += 1
        # labellistt[i * 2 + 1].config(text=len(javab[filez[0]]['number']))
        labellistt[i * 2 + 1].config(text=maindic[li[neshangar]]['number'][i])
        contor += 1
def up():
    global filez,labelList,labelListPlate,javab,Lb1,con,li
    filez = fd.askopenfilenames( title='Choose a file')
    ReadRequest.ReadReq(filez)
    javab = ReadRequest.lastdir

    for i in filez:
        li.append(i)
        Lb1.insert(tk.END, i)
        maindic[i] = javab[i]
        con += 1
    print(filez)
    changeimage()

def dellabel():
    for i in range(0,20):
        try:
            labellistt[i].config(text="")
            labellistt[i].image = None
        except:
            pass
def nextt():
    global neshangar
    print(neshangar,len(li))
    dellabel()
    if neshangar < len(li)-1:
        neshangar += 1
    changeimage()

def pree():
    global neshangar
    print(neshangar,len(li))

    dellabel()
    if neshangar!=0:
        neshangar -= 1
    changeimage()
root = Tk()
image = Image.open("ax.png")
image = image.resize((800, 800), Image.ANTIALIAS)
my_img = ImageTk.PhotoImage(image)
my_lbl = Label(image=my_img)
my_lbl.grid(row=0, column=0, rowspan=20)

labellistt = [0]*20
for i in range(0,20):
    # if i%2 == 0 :
    #     sad = Image.open("ax1.jpg")
    #     sad = sad.resize((160, 40), Image.ANTIALIAS)
    #     sad = ImageTk.PhotoImage(sad)
    #     labellistt[i] = Label()
    #     labellistt[i].image = sad
    #     labellistt[i].grid(row=i, column=1)
    # else:
    labellistt[i] = Label(text = "")
    labellistt[i].grid(row=i, column=1)

Button1 = Button(text="upload",command=up,height = 3 , width = 9)
Button1.grid(row=0, column=2)
Button2 = Button(text="Next", command=nextt,height = 3 , width = 9)
Button2.grid(row=1, column=2)
Button3 = Button(text="Previous", command=pree,height = 3 , width = 9)
Button3.grid(row=2, column=2)
Lb1 = Listbox()
Lb1.grid(row=3, column=2, rowspan=3)



root.mainloop()







# def changelabel():
#
# def changeaddress():
#
