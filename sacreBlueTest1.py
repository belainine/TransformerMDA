# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:32:06 2020

@author: belainine
"""

import sacrebleu
from sacremoses import MosesDetokenizer
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
#%matplotlib inline
np.random.seed(19680811)
md = MosesDetokenizer(lang='en')
#plt.switch_backend('agg')

# Open the test dataset human translation file and detokenize the references
refs = []

with open("eval/dailyDialog/referance.txt") as test:
    for line in test: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        refs.append(line)
    
print("Reference 1st sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions

def loadPreds(file):
    preds = []
    with open(file) as pred:  
        for line in pred: 
            line = line.strip().split() 
            line = md.detokenize(line) 
            preds.append(line)
    return preds
file=[]  
file.append("eval/dailyDialog/candidate_pred1.txt")
file.append("eval/dailyDialog/candidate_pred2.txt")
file.append("eval/dailyDialog/candidate_pred4.txt")
file.append("eval/dailyDialog/candidate_pred3.txt")
file.append("eval/dailyDialog/candidate_pred5.txt")
file.append("eval/dailyDialog/candidate_pred6.txt")
preds=[]
for i in range(6):
    preds.append(loadPreds(file[i]))



print("MTed 1st sentence:", preds[1][0]) 
# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds[0], refs)
print('Bleu of   MDA Task is:',bleu.score)

bleu = sacrebleu.corpus_bleu(preds[1], refs)
print('Bleu of HRED Task is:',bleu.score)
bleu = sacrebleu.corpus_bleu(preds[2], refs)
print('Bleu of VHRED   Task is:',bleu.score)
bleu = sacrebleu.corpus_bleu(preds[3], refs)
print('Bleu of VHCR   Task is:',bleu.score)

bleu = sacrebleu.corpus_bleu(preds[4], refs)
print('Bleu of ReCoSa  Task is:',bleu.score)

bleu = sacrebleu.corpus_bleu(preds[5], refs)
print('Bleu of HRAN  Task is:',bleu.score)
nb_iteration=100
n_size = int(len(preds[1]) * 0.50)
bleus=[[] for i in range(6)]
preds_=['' for j in range(6)]
for i in range(nb_iteration):
    
    inds = resample([i for i in range(len(preds[1]))], n_samples=n_size)
    refs_1=[[ refs[0][i] for i in inds]]
    for j in range(6):
        
        preds_[j]= [ preds[j][k] for k in inds]
        bleu = sacrebleu.corpus_bleu(preds_[j], refs_1)
        bleus[j].append(bleu.score)

    #print('bleu',bleus[0][-1],bleus[1][-1])
    
#plt.hist(bleus[0])
#plt.figure(figsize = (10,nb_iteration))
    
#plt.hist(bleus[1])
#plt.figure(figsize = (10,nb_iteration))
#Lets find Confidence intervals

a = 0.95 # for 95% confidence



def ConfidenceIntervals(stats):
    p = ((1.0 - a)/2.0) * 100 #tail regions on right and left .25 on each side indicated by P value (border)
                          # 1.0 is total area of this curve, 2.0 is actually .025 thats the space we would want to be 
                            #left on either side=
    lower = max(0.0, np.percentile(stats,p))
    
    p = (a + ((1.0 - a)/ 2.0)) * 100 #p is limits
    upper = min(100.0, np.percentile(stats,p))
    return lower,upper
lower=['' for i in range(6)]
upper=['' for i in range(6)]

for i in range(6):
    stats=bleus[i]
    lower[i], upper[i]=ConfidenceIntervals(stats)
    print('%.1f confidence interval [%.1f%% and %.1f%%]' %(a*100, lower[i], upper[i]))
    print(' Mean and Std [%.1f and %.1f]' %( np.mean(stats), np.std(stats)))


def millions(x):
    return '$%1.1fM' % (x*1e-6)


# Fixing random state for reproducibility

plt.figure(figsize=(100,100))
x = np.arange(0,len(bleus[1]))

y0 = np.array(bleus[1])
y1 = np.array(bleus[0])
y2 = np.array(bleus[2])
y3 = np.array(bleus[3])
y4 = np.array(bleus[4])
y5 = np.array(bleus[5])
fig, ax = plt.subplots()
ax.fmt_ydata = millions
plt.plot(x,y1, color='b',label='MDATransformer resampling')
plt.plot(x, y4, color='r',label=' ReCoSa resampling')
plt.plot(x, y5, color='Orange',label=' HRAN resampling')
plt.plot(x, y2,color='g',label='VHCR resampling')
plt.plot(x, y3,color='m',label='VHRED resampling')
plt.plot(x, y0,color='k',label='HRED resampling')
#plt.axhline(y=lower1, color='r', linestyle='-.', label='95% BLEU interval DSM')
#plt.axhline(y=upper1, color='r', linestyle='-.')
#plt.axhline(y=lower2, color='b', linestyle='--', label='95% BLEU interval Trans')
#plt.axhline(y=upper2, color='b', linestyle='--')
plt.axhline(y=np.mean(y4), color='r')#,label='Mean  BLEU of MDATransformer')
plt.axhline(y=np.mean(y5), color='Orange')#,label='Mean  BLEU of MDATransformer')
plt.axhline(y=np.mean(y1), color='b')#,label='Mean  BLEU of ReCoSa')

plt.axhline(y=np.mean(y2), color='g')#,label='Mean  BLEU of HRED')

plt.axhline(y=np.mean(y3), color='m')#,label='Mean  BLEU of VHRED')

plt.axhline(y=np.mean(y0), color='k')#,label='Mean  BLEU of VHCR')
plt.title("Statistical Significance")
plt.xlabel("Iteration number")
plt.ylabel("BLEU");
plt.xlim((0, 100))
plt.ylim((0, 6))
plt.legend(loc=0, prop={'size': 10})

plt.show()

# Fixing deferances

plt.figure(figsize=(100,100))
y0 = np.array(bleus[0])-np.array(bleus[1])
y1 = np.array(bleus[0])-np.array(bleus[2])
y2 = np.array(bleus[0])-np.array(bleus[3])
y3 = np.array(bleus[0])-np.array(bleus[4])
y4 = np.array(bleus[0])-np.array(bleus[5])
x = np.arange(0,len(bleus[1]))

fig, ax = plt.subplots()
ax.fmt_ydata = millions
plt.plot(x, y0,label='$\Delta$BLEU(MDA-HRED)')
plt.plot(x, y1,label='$\Delta$BLEU(MDA-VHCR)')
plt.plot(x, y2,label='$\Delta$BLEU(MDA-VHRED)')
plt.plot(x, y3,label='$\Delta$BLEU(MDA-ReCoSa)')
plt.plot(x, y4,label='$\Delta$BLEU(MDA-HRAN)')
plt.title("The difference between experiences")
plt.xlabel("Iteration number")
plt.ylabel(" BLEU MDA - BLEU Other");
plt.xlim((0, 100))
plt.ylim((0, 5))
plt.legend(loc=0, prop={'size': 10})

plt.show()
