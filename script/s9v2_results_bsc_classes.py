# PYTHON 3
import os
import pandas as pd
import numpy as np
import re
import glob
import sys
from matplotlib import pyplot

StromaVSautre = '../results/StromaVSautre_DB10/LamesCompletes/Stroma/'
TumorVSNormal = '../results/StromaVSautre_DB10/LamesCompletes/N_T/'
xy_path = '/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/scan_tiles/'
out = '../results/StromaVSautre_DB10/LamesCompletes/CombModels_BSC_prediction/'


def process_df(myPred):
    truth, name_tile, pred_tile = [[] for i in range(3)]
    myPred.rename(columns={0:'V1', 1:'V2', 2:'V3', 3:'V4'}, inplace=True)
    for row in myPred['V1']:
        truth.append(row.split('/')[0])
        name_tile.append(row.split('/')[1])
    myPred['truth'] = truth
    myPred['name_tile'] = name_tile

def generate_pred_tile(myPred):
    myPred['pred_tile'] = None
    for i, val in enumerate(myPred['pred_tile']):
        if myPred['V2'][i]>0.5:
            myPred.loc[i, 'pred_tile'] = 'Duodenum'
        else:
            myPred.loc[i, 'pred_tile'] = np.nan

        if myPred['V3'][i]>0.5:
            myPred.loc[i, 'pred_tile'] = 'N_T'
        else:
            myPred.loc[i, 'pred_tile'] = myPred.loc[i, 'pred_tile']

        if myPred['V4'][i]>0.5:
            myPred.loc[i, 'pred_tile'] = 'Stroma'
        else:
            myPred.loc[i, 'pred_tile'] = myPred.loc[i, 'pred_tile']

def generate_classes(myPred):
    myPred['classes'] = None

    for i, val in enumerate(myPred['classes']):
        if myPred['V4'][i]>0.8:
            myPred.loc[i, 'classes'] = 'classe3'
        else:
            myPred.loc[i, 'classes'] = np.nan

        if myPred['V4'][i]>0.5 and myPred['V4'][i]<0.8:
            myPred.loc[i, 'classes'] = 'classe4'
        else:
            myPred.loc[i, 'classes'] = myPred.loc[i, 'classes']

        if myPred['V2'][i]>0.8:
            myPred.loc[i, 'classes'] = 'classe7'
        else:
            myPred.loc[i, 'classes'] = myPred.loc[i, 'classes']

        if myPred['V2'][i]>0.5 and myPred['V2'][i]<0.8 :
            myPred.loc[i, 'classes'] = 'classe8'
        else:
            myPred.loc[i, 'classes'] = myPred.loc[i, 'classes']

def generate_pred_class_NvsT(myPred):
    myPred['pred_class_NvsT'] = None
    for i, val in enumerate(myPred['pred_class_NvsT']):
        if myPred['V2'][i]>0.5:
            myPred.loc[i, 'pred_class_NvsT'] = 'Normal'
        else:
            myPred.loc[i, 'pred_class_NvsT'] = 'Tumor'

def generate_classesTvsNT(myPred):
    myPred['classesTvsNT'] = None
    for i, val in enumerate(myPred['classesTvsNT']):
        if myPred['V2'][i]>0.8:
            myPred.loc[i, 'classesTvsNT'] = 'classe1'
        else:
            myPred.loc[i, 'classesTvsNT'] = np.nan

        if myPred['V2'][i]>0.5 and myPred['V2'][i]<0.8:
            myPred.loc[i, 'classesTvsNT'] = 'classe2'
        else:
            myPred.loc[i, 'classesTvsNT'] = myPred.loc[i, 'classesTvsNT']

        if myPred['V3'][i]>0.8:
            myPred.loc[i, 'classesTvsNT'] = 'classe5'
        else:
            myPred.loc[i, 'classesTvsNT'] = myPred.loc[i, 'classesTvsNT']

        if myPred['V3'][i]>0.5 and myPred['V3'][i]<0.8 :
            myPred.loc[i, 'classesTvsNT'] = 'classe6'
        else:
            myPred.loc[i, 'classesTvsNT'] = myPred.loc[i, 'classesTvsNT']


# MAIN
if __name__ == '__main__':

    arg = sys.argv[1] #MY SLIDE ID
    myPrediction = pd.read_csv(f'{StromaVSautre}myPrediction_{arg}.csv',sep=',', header=None)
    process_df(myPrediction)
    generate_pred_tile(myPrediction)
    generate_classes(myPrediction)
    myPrediction_TvsNT = pd.read_csv(f'{TumorVSNormal}myPrediction_{arg}.csv',sep=',', header=None)
    process_df(myPrediction_TvsNT)
    generate_pred_class_NvsT(myPrediction_TvsNT)
    generate_classesTvsNT(myPrediction_TvsNT)

    m1 = myPrediction[['name_tile', 'truth', 'pred_tile', 'classes']]
    m2 = myPrediction_TvsNT[['name_tile', 'pred_class_NvsT', 'classesTvsNT']]
    myPred = pd.merge(m2, m1, on='name_tile', how='right')

    # remove '.tif'
    for ind, val in enumerate(myPred['name_tile']):
        myPred.loc[ind, 'name_tile'] = val.split('.')[0]

    for i, val in enumerate(myPred['classes']):
        if val is np.nan:
            myPred.loc[i, 'classes'] = myPred.loc[i, 'classesTvsNT']

    myPred = myPred[['name_tile', 'classes']].rename(columns={'name_tile':'tuile', 'classes':'classPath'})

    tileXY = pd.read_csv(f'{xy_path}{arg}/{arg}_tileXY.txt', sep='\t', header=0)

    temp = []
    for i in tileXY['Tile-Point']:
        temp.append(i.split(' ')[1])
    tileXY['Point'] = temp
    tileXY = tileXY[tileXY['Point']=='Point1'].drop('Point', axis=1)
    temp = []
    for i in tileXY['Tile-Point']:
        temp.append(re.sub('Tile', '', i).split(' ')[0])
    tileXY['Tile-Point'] = temp
    temp = []
    for i in tileXY['Tile-Point']:
        temp.append(f'{arg}_{i}')
    tileXY['tuile'] = temp

    myData = pd.merge(myPred, tileXY, on ='tuile', how='right')

    myData['Tile-Point']= myData['Tile-Point'].astype('int')
    myData.sort_values(by='Tile-Point', inplace=True)
    myData = myData[["tuile","classPath"]]
    myData['classPath'].value_counts()

    pyplot.figure(figsize=(10,10))
    myData['classPath'].value_counts().plot.pie(explode = [0.1, 0, 0, 0, 0, 0, 0, 0],
                                            autopct='%.2f',
                                           shadow = True)
    pyplot.title(arg)
    pyplot.ylabel('')

    pyplot.savefig(f'../results/StromaVSautre_DB10/LamesCompletes/pie_classes_distribution/{arg}.jpeg')
    myData.to_csv(f'{out}{arg}.csv', index=False, na_rep=np.nan)
