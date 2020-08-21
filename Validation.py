from sklearn import linear_model
import numpy as np
import pandas as pd
import glob
import scipy
import matplotlib.pyplot as plt


clf = linear_model.LinearRegression()
std = linear_model.LinearRegression()

def mono_reg(explanation,objective):
    X = np.array(explanation)
    Y = np.array(objective)

    Y = Y.reshape(-1, 1)
    X = X.reshape(-1, 1)

    clf.fit(X, Y)

    print(clf.coef_)
    print(clf.intercept_)
    print(clf.score(X, Y))

def corr(All_suspect, Hokkaido,A,B):
    corr = np.correlate(All_suspect, Hokkaido, "full")/len(All_suspect)
    print(np.amax(corr))
    estimated_delay = corr.argmax() - (len(News) - 1)
    print("estimated delay is " + str(estimated_delay))
    plt.subplot(4, 1, 1)
    plt.ylabel(A)
    plt.plot(All_suspect)

    plt.subplot(4, 1, 2)
    plt.ylabel(B)
    plt.plot(Hokkaido, color="g")

    plt.subplot(4, 1, 3)
    plt.ylabel("fit")
    plt.plot(np.arange(len(All_suspect)), All_suspect)
    plt.plot(np.arange(len(Hokkaido)) + estimated_delay, Hokkaido)
    plt.xlim([0, len(All_suspect)])

    plt.subplot(4, 1, 4)
    plt.ylabel("corr")
    plt.plot(np.arange(len(corr)) - len(Hokkaido) + 1, corr, color="r")
    plt.xlim([-len(All_suspect)//2, len(All_suspect)//2])

    plt.show()

def linear_std(X,Y):

    xss_sk = scipy.stats.zscore(X)
    yss_sk = scipy.stats.zscore(Y)

    return xss_sk,yss_sk

def population_dict():
    Area_csv=[]
    all_area = glob.glob('./Hokkaido/*.txt')
    for filename in all_area:
        Area_csv.append(pd.read_csv(filename,skiprows=[1],encoding='SHIFT_JIS'))
    df = pd.concat(Area_csv)
    POP = df[['KEY_CODE','T000847001']].astype(int).rename(columns={'T000847001':'Population'})

    POP['KEY_CODE'] =(POP['KEY_CODE']/10).astype(int)
    Population = POP.groupby(['KEY_CODE']).sum()
    Population_todict = Population.to_dict()
    Population_dict = Population_todict['Population']

    return Population_dict


all_files = glob.glob("./20200316/pros0123-0305/*.txt")
all_files.sort()
csvs = []
c_name = []
for filename in all_files:
    readtables=pd.read_table(filename,header = None,usecols=[0,2])
    readt = readtables.astype(int)
    grouped=readt.groupby(0).sum()
    csvs.append(grouped)
    c_name.append(filename.replace('./20200316/pros0123-0305/cov_', '').replace('_hokkaido_pros.txt', ''))
df = pd.concat(csvs,axis=1)
df.sort_index()
df.columns = c_name
df2=df.fillna(0)
Hokkaido = pd.Series([0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,1, 0,0,0,0,2,1,3, 9,7,6,5,8,15,12, 4,2,5,2,3,1,])
Ishikari = pd.Series([0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,1, 0,0,0,0,1,1,1, 1,2,3,1,1,2,4, 3,1,1,2,0,1,])
Ohotuku  = pd.Series([0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,0,1,1,1,3,2, 0,1,3,0,1,0,])
Other    = pd.Series([0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 0,0,0,0,2,0,2, 7,5,0,3,6,10,6, 1,0,1,0,2,0,])
News =     pd.Series([8,21,9,3,14,33,27,15,20, 5,7,6,32,16,12,17, 6,4,9,8,19,13,52, 34,17,36,36,77,76,141, 118,78,87,161,223,261,210, 172,147,129,30+91+65,31+93+90,7+98+51])


All_suspect = df2.sum()
Suspect_7d = []
Infect_7d = []

df_1 = pd.read_csv('./meshcity/01-1.csv',encoding = "shift-jis")
df_2 = pd.read_csv('./meshcity/01-2.csv',encoding = "shift-jis")
df_3 = pd.read_csv('./meshcity/01-3.csv',encoding = "shift-jis")

meshcity = pd.concat([df_1,df_2,df_3])

ishikari = []
ohotuku = []
others = []
for index, row in meshcity.iterrows():
    if row[0] == 1101 or row[0] == 1102 or row[0] == 1103 or row[0] == 1104 or row[0] == 1105 or row[0] == 1106\
       or row[0] == 1107 or row[0] == 1108 or row[0] == 1109 or row[0] == 1110 or row[0] == 1217 or row[0] == 1234\
            or row[0] == 1235 or row[0] == 1303 or row[0] == 1304 or row[0] == 1224 or row[0] == 1231:
        ishikari.append(row[2])
    elif row[0] == 1543 or row[0] == 1544 or row[0] == 1564 or row[0] == 1545 or row[0] == 1546 or row[0] == 1547\
            or row[0] == 1549 or row[0] == 1550 or row[0] == 1552 or row[0] == 1555 or row[0] == 1559 or row[0] == 1560\
            or row[0] == 1561 or row[0] == 1562 or row[0] == 1563 or row[0] == 1211 or row[0] == 1208 or row[0] == 1219:
        ohotuku.append(row[2])
    else:
        others.append(row[2])

ishikari_DF = pd.DataFrame()
ohotuku_DF = pd.DataFrame()
others_DF = pd.DataFrame()

for index_name, item in df2.iterrows():
    if index_name in ishikari:
        ishikari_DF[index_name] = item
    elif index_name in ohotuku:
        ohotuku_DF[index_name] = item
    elif index_name in others:
        others_DF[index_name] = item

ishikari_suspect = ishikari_DF.sum(axis=1)
ohotuku_suspect = ohotuku_DF.sum(axis=1)
others_suspect = others_DF.sum(axis=1)

if __name__ == '__main__':

    AAA , BBB = linear_std(ohotuku_suspect,Ohotuku)
    corr(AAA,BBB,'Ishikari_suspect','Ishikari_cases')
