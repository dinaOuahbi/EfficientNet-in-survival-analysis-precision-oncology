# python 0
# DO

# obj / select tiles that were predicted as normal or tumoral previously with densenet

# SELECTIONNER LES TUILES PREDITE COMME NORMAL OU TUMORAL
import pandas as pd
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import progressbar
import glob
import time
import sys

slide = sys.argv[1]
'''slide = slide.split('.')[0]'''

'''if slide in os.listdir('/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/Results/SCNN/N_Tselect/'):
    print('DONE ')
    sys.exit()'''

snn = '../results/'
SCAN_reinhard = '/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/SCAN_reinhard/'
svsa = f'{snn}StromaVSautre_DB10/LamesCompletes/Stroma/'

def select_tiles_NT(slide):
    myPrediction = pd.read_csv(f'{svsa}myPrediction_{slide}.csv', sep=',', header=None)
    myPrediction.rename(columns={0:'V1', 1:'V2', 2:'V3', 3:'V4'}, inplace=True)
    truth, name_tile, pred_tile = [[] for i in range(3)]
    for row in myPrediction['V1']:
        truth.append(row.split('/')[0])
        name_tile.append(row.split('/')[1])
    myPrediction['truth'] = truth
    myPrediction['name_tile'] = name_tile

    myPrediction['pred_tile'] = None
    for i, val in enumerate(myPrediction['pred_tile']):
        if myPrediction['V2'][i]>0.5:
            myPrediction.loc[i, 'pred_tile'] = 'Duodenum'

        elif myPrediction['V3'][i]>0.5:
            myPrediction.loc[i, 'pred_tile'] = 'N_T'

        elif myPrediction['V4'][i]>0.5:
            myPrediction.loc[i, 'pred_tile'] = 'Stroma'
        else:
            myPrediction.loc[i, 'pred_tile'] = np.nan

    try:
        os.mkdir(f'{snn}N_Tselect')
        os.mkdir(f'{snn}N_Tselect/pred_class_dist_png')
    except FileExistsError as err:
        print(err)
        pass
    os.mkdir(f'{snn}N_Tselect/{slide}')
    plt.figure(figsize=(10, 5))
    myPrediction['pred_tile'].value_counts().plot(kind='bar', color='cyan', edgecolor='red')
    for i in range(myPrediction['pred_tile'].value_counts().shape[0]):
        plt.annotate(myPrediction['pred_tile'].value_counts()[i],
                     (-0.1+i,myPrediction['pred_tile'].value_counts()[i]+10))
    plt.legend(labels=['Total tiles'])
    plt.title('Pred classes distribution')
    plt.grid(True)
    plt.savefig(f'{snn}N_Tselect/pred_class_dist_png/Pred_classes_distribution_{slide}.png')

    mySelect = myPrediction['V1'][myPrediction['pred_tile']=='N_T']
    bar = progressbar.ProgressBar(maxval=len(mySelect),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i, select in enumerate(mySelect):
        #name_tile = myPrediction['name_tile'][myPrediction['V1']==select]
        ORG = f"{SCAN_reinhard}{slide}/{select.split('/')[1]}"
        TARGET = f"{snn}N_Tselect/{slide}/{select.split('/')[1]}"
        shutil.copyfile(ORG, TARGET) #/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/SCAN_reinhard/myPrediction_21_0_005_00_13/21_0_005_00_13_10046.tif'
        bar.update(i)
    bar.finish()

if __name__ == '__main__':
    print(slide)
    select_tiles_NT(slide)
