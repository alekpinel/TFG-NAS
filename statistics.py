import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import ReadData, CalculateAccuracy, extraerSP_SS, convertToBinary

#Plot bars. Data must be ("title", value) or ("title", value, color)
def PlotBars(data, title=None, y_label=None, dateformat=True):
    strings = [i[0] for i in data]
    x = [i for i in range(len(data))]
    y = [i[1] for i in data]
    
    colors=None
    if (len(data[0]) > 2):
        colors = [i[2] for i in data]
    
    fig, ax = plt.subplots()
    
    if (title is not None):
        ax.set_title(title)
    if (y_label is not None):
        ax.set_ylabel(y_label)
    
    if (dateformat):
        fig.autofmt_xdate()
        
    x_labels=strings
    plt.xticks(x, x_labels)
    
    if (colors is not None):
        plt.bar(x, y, color=colors)
    else:
        plt.bar(x, y)
    plt.show()

def class_percentages(Y):
    unique, counts = np.unique(Y, return_counts=True)
    total = sum(counts)
    if (len(unique) == 3):
        percents = [x/total*100 for x in counts]
        data = [("Adenoma", percents[0], 'blue'), ("Hyperplasic", percents[1], 'green'), 
                ("Serrated", percents[2], 'red')]
        print(f"Percent of each polyp:\nAdenoma: {percents[0]}\nHyperplasic: {percents[1]}\nSerrated {percents[2]}")
    else:
        percents = [x/total*100 for x in counts]
        data = [("Reject", percents[0], 'red'), ("No Reject", percents[1], 'blue')]
        print(f"Percent of each polyp:\nReject: {percents[0]}\nNo Reject: {percents[1]}")
    
    PlotBars(data, "Class Percentages", "Percent", dateformat=False)

def main():
    print("Statistics")
    
    X, Y = ReadData(light='NBI')
    class_percentages(Y)
    Y = convertToBinary(Y)
    class_percentages(Y)
    
    
    
    
    

if __name__ == '__main__':
  main()