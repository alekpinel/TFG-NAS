import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import ReadData, CalculateAccuracy, extraerSP_SS, convertToBinary, calculate_metrics

#Plot bars. Data must be ("title", value) or ("title", value, color)
def PlotBars(data, title=None, y_label=None, dateformat=False):
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
        data = [("Adenomas", percents[0], 'blue'), ("Lesiones hiperplásicas", percents[1], 'green'), 
                ("Pólipos serrados", percents[2], 'red')]
        print(f"Number of each polyp:\nAdenoma: {counts[0]}\nHyperplasic: {counts[1]}\nSerrated {counts[2]}\nTotal {total}")
        print(f"Percent of each polyp:\nAdenoma: {percents[0]}\nHyperplasic: {percents[1]}\nSerrated {percents[2]}")
    else:
        percents = [x/total*100 for x in counts]
        data = [("Resection", percents[0], 'red'), ("No Resection", percents[1], 'blue')]
        print(f"Percent of each polyp:\nReject: {percents[0]}\nNo Reject: {percents[1]}")
    
    PlotBars(data, "Distribución de clases", "Porcentaje")


def print_result(results, key, title='Results'):
    data = []
    for result in results:
        data.append((result['name'], result[key]))
    PlotBars(data, title, "Percent")
    
def print_metrics(tp, fn, fp, tn):
    accuracy, specificity, sensitivity, precision, f1score = calculate_metrics(tp, fn, fp, tn)
    print(f"\nRESULTS: \nACC:{accuracy:.3f} \nSS:{sensitivity:.3f} \nSP:{specificity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f}")

def main():
    print("Statistics")
    
    print_metrics(15, 2, 2, 4)
    return
    
    X, Y = ReadData(light='NBI')
    class_percentages(Y)
    Y = convertToBinary(Y)
    class_percentages(Y)
    
    results = []
    results.append( {'name':'Diseño Experto',
             'ACC':0.739,
             'SS':0.762,
             'SP':0.500,
             'Pr':0.941,
             'Score':0.740,
             'Time':24.510,
        })
    
    
    

if __name__ == '__main__':
  main()