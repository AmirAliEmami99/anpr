import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image


def Page(nameOfFile,Tedad,PN,pelaks):
    # root = Tk()
    # logo = tk.PhotoImage(file="Pelaks/"+nameOfFile+"/directory.png")
    # w1 = tk.Label(root, image=logo, width=600, height=500).pack(side="left")
    # root.mainloop()
    root = Tk()

    label1 = Label(text="Addresses")
    label1.grid(row=0, column=2,rowspan=1)
    Button1 = Button(text="upload",height = 3, width = 20,command = up)
    Button1.grid(row=1, column=2)

    # root.columnconfigure(0, weight=6)
    # root.columnconfigure(1, weight=1)

    # image = Image.open("Pelaks/"+nameOfFile+"/directory.png")
    image = Image.fromarray(nameOfFile)
    image = image.resize((600, 600), Image.ANTIALIAS)
    my_img = ImageTk.PhotoImage(image)
    my_lbl = Label(image=my_img)
    my_lbl.grid(row=0, column=0,rowspan=Tedad*2 ,columnspan =Tedad*2)
    labelList = [0]*Tedad
    labelListPlate = [0]*Tedad
    imageList = [0]*Tedad
    shamorti  = [0]*Tedad
    for i in range(0,Tedad):
        # root = Tk()
        # imageList[i] = Image.open("Pelaks/" + nameOfFile +"/"+ str(i+1)+"/Pelak"+str(i+1)+".png")
        imageList[i] = Image.fromarray(pelaks[i])
        imageList[i] = imageList[i].resize((200, 60), Image.ANTIALIAS)
        shamorti[i] = ImageTk.PhotoImage(imageList[i])
        labelList[i] = Label(image=shamorti[i])
        labelList[i].grid(row=i*2, column=Tedad*2+1)
        labelListPlate[i] = Label(text = PN[i])
        labelListPlate[i].grid(row=i*2+1, column=Tedad*2+1)
    root.mainloop()

# if __name__ == '__main__':
# Page("2021-01-15_14-15-42-534_21M87888-IRN.jpg",2,["sdssdsdsd","123234"])