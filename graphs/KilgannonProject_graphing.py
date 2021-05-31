"""
 
CISC 642
Final Project
Graphing
Jon Kilgannon
 
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

network_num = 6

file_name = "output" + str(network_num) + ".txt"

f = open(file_name, "r")
prevline = ""

train_losses = []
val_losses = []
accs = []
val_accs = []
epochs = []

for line in f:
    if line[10:15] == "=====":
        # We are on the right line
        epochtext = prevline[6:9]
        
        buildepoch = ""
        for ch in epochtext:
            if ch == "/":
                break
            else:
                buildepoch = buildepoch + ch
        
        epoch = int(buildepoch)
        
        loss_start = line.find(" loss:")
        val_loss_start = line.find("val_loss:")
        acc_start =  line.find(" accuracy:")
        val_acc_start =  line.find("val_accuracy:")
                
        train_loss = float(line[(loss_start+7) : (loss_start+13)])
        val_loss = float(line[(val_loss_start+10) : (val_loss_start+16)])
        acc = float(line[(acc_start+11) : (acc_start+17)])
        val_acc = float(line[(val_acc_start+14) : (val_acc_start+20)])

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accs.append(acc)
        val_accs.append(val_acc)
    
    prevline = line     

# Now make a graph
fig,ax = plt.subplots()

ax.plot(epochs, train_losses, label = "Training Loss", color="red")
ax.plot(epochs, val_losses, label = "Validation Loss", color="blue")
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

ax2=ax.twinx()
ax2.plot(epochs, accs, label='Training Accuracy', color="orange")
ax2.plot(epochs, val_accs, label='Validation Accuracy', color="black")
ax2.set_ylabel('Accuracy')

plt.xlabel('Epochs')
plt.title('Training and Validation Loss and Accuracy for Network ' + str(network_num))
ax.legend(loc=2)
ax2.legend(loc=9)
plt.show()

