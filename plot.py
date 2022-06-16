import numpy as np
import matplotlib.pyplot as plt

def plot_label_corruptions(label_accs):
    markers = ['x','o','v','*','^','s']
    
    for opt in label_accs:
        #Not enough lines, or too many
        if len(label_accs[opt]) != len(markers):
            continue
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        #Plot label accuracies
        for acc, marker in zip(label_accs[opt], markers):
            plt.plot(range(1,len(acc)+1,1),acc, marker=marker, linewidth=2.5)
        
        plt.legend([f'{x}% label corruption' for x in [0, 20, 40, 60, 80, 100]])
        plt.ylabel('Train accuracy (%)')
        plt.xlabel('Epoch')
        #Add '%' symbol to yticks
        plt.yticks(range(10,110,10), [f'{x}%' for x in range(10,110,10)])
        plt.title(opt)
        plt.xlim([0,35])
        #Force plot to be square
        ax.set_aspect(1./ax.get_data_ratio())
        plt.savefig(f'{opt}_corrupt_labels.pdf')
    

def plot_image_corruptions(image_accs):
    markers = markers = ['x','o','*', '^', 's']
    for opt in image_accs:
        #Not enough lines, or too many
        if len(image_accs[opt]) != len(markers):
            continue
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        for acc, marker in zip(image_accs[opt], markers):
            plt.plot(range(1, len(acc)+1), acc, marker=marker,linewidth=2.5)
        plt.legend(['True labels','Shuffling along\nx-axis', 'Shuffling along\nx and y-axis', 'Gaussian noise', '100% random labels'])
        plt.ylabel('Train accuracy (%)')
        plt.xlabel('Epoch')
        #Add '%' symbol to yticks
        plt.yticks(range(10,110,10), [f'{x}%' for x in range(10,110,10)])
        plt.title(opt)
        #Force plot to be square
        ax.set_aspect(1./ax.get_data_ratio())

        plt.savefig(f'{opt}_corrupt_images.pdf')

def save_plots(label_accs, image_accs):
    plot_label_corruptions(label_accs)
    plot_image_corruptions(image_accs)