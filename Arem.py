
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, label_binarize
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import seaborn as sns; sns.set(style="white", color_codes=True)
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import bootstrapped.stats_functions as bs_stats
from sklearn.feature_selection import RFE
import bootstrapped.bootstrap as bs
from prettytable import PrettyTable
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats as stat
import pandas as pd
import numpy as np
# %matplotlib inline
import os, re, sys
import warnings

warnings.filterwarnings('ignore')

# Cloning Git Repo
! git clone https://github.com/adityasamant/AreM-Time-Series-Classification


# Lists of Data (META DATA)
column_name = ['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23']
activity_name = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
test_name = ['bending1/dataset1.csv', 'bending1/dataset2.csv', 
            'bending2/dataset1.csv', 'bending2/dataset2.csv', 
            'cycling/dataset1.csv', 'cycling/dataset2.csv', 'cycling/dataset3.csv',
            'lying/dataset1.csv', 'lying/dataset2.csv', 'lying/dataset3.csv',
            'sitting/dataset1.csv', 'sitting/dataset2.csv', 'sitting/dataset3.csv',
            'standing/dataset1.csv', 'standing/dataset2.csv', 'standing/dataset3.csv',
            'walking/dataset1.csv', 'walking/dataset2.csv', 'walking/dataset3.csv' ]
activity_instance = [7,6,15,15,15,15,15]
test_instance = [2,2,3,3,3,3,3]
train_instance = [5,4,12,12,12,12,12]
dflist = []
testlist = []
trainlist = []

# Sorting Function for alphanumeric data
def sorted_an(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# Reading Excel Sheet
for root, dirs, files in sorted(os.walk('AReM/')):
    for fname in sorted_an(files):
        if re.match("^.*.csv$", fname):
            df = pd.read_csv(os.path.join(root, fname), skiprows=5, sep='[,, ]', names=column_name, usecols=range(7))
            for y in df.columns:
                if not (df[y].dtype == np.float64 or df[y].dtype == np.int64):
                    print("Data not correct")
                    break
            name = root[17:]+"/"+fname
            if(name in test_name):
                testlist.append(df)
            else:
                trainlist.append(df)
            dflist.append(df)

# Train Test Data Size
print("Test Data is of size",len(testlist),"and Train Data is of size",len(trainlist))

# showing random Data
dflist[87].head(5)



def createD(dflist,divide):
    
    # new data column list
    new_names = []
    for i in range(1,6*divide+1):
        new_names.extend(["Min "+str(i),"Max "+str(i),"Mean "+str(i),"Med "+str(i),"Std "+str(i),"1st "+str(i),"3rd "+str(i)])
    
    # Extracting features from data and creating new data
    datalist = []
    for dsfull in dflist:
        temp = []
        location=[-1]
        num = len(dsfull)
        den = divide
        for l in range(divide):
            ans = int(num/den)
            location.append(location[l]+ans)
            num = num - ans
            den = den - 1
        
        dslist = [dsfull.iloc[location[d]+1:location[d+1]] for d in range(divide)]
        for ds in dslist:
            dmin = ds.min() # minimun
            dmax = ds.max() # Maximum
            dmean = ds.mean() # Mean
            dmedian = ds.median() # Median
            dstd = ds.std() # Standard Deviation
            d1q = ds.quantile(0.25) # First Quantile
            d3q = ds.quantile(0.75) # Third Quantile
            for index in range(1,7):
                temp.extend([dmin[index],dmax[index],dmean[index],dmedian[index],dstd[index],d1q[index],d3q[index]])
        datalist.append(temp)
        
    return pd.DataFrame(datalist, columns=new_names)

# creating DataFrame
data = createD(dflist,1)
datatr = createD(trainlist,1)
datate = createD(testlist,1)

# print shape
print("Shape of data is",data.shape)

# showing New Data
data.head(5)


# Standardization method
def NSE(datatc,datatf):
    sc = MinMaxScaler()
    # sc = StandardScaler()
    sc.fit(datatf)
    return pd.DataFrame(sc.transform(datatc),columns=datatc.columns)


Ndata = NSE(data,data)
Ndatatr = NSE(datatr,datatr)
Ndatate = NSE(datate,datatr)

# showing Standardized data
Ndata.head(5)


# Estimation of Standard Deviations
print("As the data is standardized, my estimate for std dev is 0.2")
print("\nSTD DEV of Raw Data\n{}".format(data.std()[:5]))
print("\nSTD DEV of Normalized Data\n{}".format(Ndata.std()[:5]))

# Calculating Standard Deviation
ciM = []
for name in Ndata.columns:
    ci = bs.bootstrap(np.array(data[name]), stat_func=bs_stats.std, alpha = 0.1)
    ciM.append([name,ci.lower_bound,ci.upper_bound])
    
# Creating Output DataFrame and showing it.    
ciD = pd.DataFrame(ciM, columns=["Feature","Lower Bound","Upper Bound"])
print("90% bootsrap confidence interval for the standard deviation of each feature.")
ciD



# creating target variable for training data
d_targettr = []
for l in range(len(Ndatatr)):
    if l < 9:
        d_targettr.append(1)
    else:
        d_targettr.append(0)
d_Ndatatr = Ndatatr.copy()
d_Ndatatr['target']=d_targettr
 
# creating target variable for testing data
d_targette = []
for l in range(len(Ndatate)):
    if l < 4:
        d_targette.append(1)
    else:
        d_targette.append(0)
d_Ndatate = Ndatate.copy()
d_Ndatate['target']=d_targette
d_datatr = datatr.copy()
d_datatr["target"]=d_targettr


def scatterplot(dataset,features,data):
    colname = []
    olname = []
    for ds in dataset:
        for feature in features:
            colname.append(feature+ds)
            olname.append("")
            olname.append(feature+ds)

    # Diagonal Function
    i = 0
    def diagfunc(x, **kws):
        nonlocal i
        ax = plt.gca()
        ax.annotate(olname[i], xy=(0.5, 0.5), xycoords=ax.transAxes)
        i = i+1

    # Using seaborn pairgrid to create scatterplot matrix
    scatplot = sns.PairGrid(data, hue='target',palette=["#072960","#ff0004"],vars=colname)
    scatplot = scatplot.map_offdiag(plt.scatter,edgecolor="k",s=40)
    scatplot = scatplot.map_diag(diagfunc)
    scatplot = scatplot.add_legend()

    for axis in scatplot.axes.flatten():
        axis.set_xlabel("")
        axis.set_ylabel("")
        
# I drew scatterplot on series 1 2 6 and on 3 4 5 and found the later to be better.        
scatterplot([' 3',' 4',' 5'],['Min','Max','Mean'],d_datatr)



d2_data = createD(dflist,2)
d2_datatr = createD(trainlist,2)
d2_Ndata = NSE(d2_data,d2_data)
d2_Ndatatr = NSE(d2_datatr,d2_datatr)
d2_Ndatatr['target']=d_targettr
d2_datatr['target']=d_targettr
d2_datatr.head(5)

scatterplot([' 1',' 2',' 12'],['Min','Max','Mean'],d2_datatr)



bestl = []
for l in range(1,21):
    # creating DataFrame of training data
    d3_datatr = createD(trainlist,l)
    d3_datate = createD(testlist,l)
    
    # Normalizing Data
    d3_Ndatatr = NSE(d3_datatr,createD(dflist,l))
    d3_Ndatate = NSE(d3_datate,createD(dflist,l))

    # Logistic Regression Model Created
    LRmodeld3 = LogisticRegression(C = 1e30, n_jobs=-1)
    LRmodeld3.fit(d3_Ndatatr,d_targettr)
        
    # Running Cross Validation
    score = LRmodeld3.score(d3_Ndatate,d_targette)   
    bestl.append(score)
print("Accuracy of LR's for L : {1,2....20} is",bestl[:5],"and so on")

bestlp = []
for l in range(1,21):
    # creating DataFrame of training data
    d3_datatr = createD(trainlist,l)
    
    # Normalizing Data
    d3_Ndatatr = NSE(d3_datatr,createD(dflist,l))
    
    # For every p value
    for p in range(1,len(d3_Ndatatr.columns)+1):
        
        # Logistic Regression Model Created
        LRmodellp = LogisticRegression(C = 1e30, n_jobs=-1)
        
        # select p features using recursive feature selection
        rfe = RFE(LRmodellp, p)
        rfe = rfe.fit(d3_Ndatatr,d_targettr)
        selector = rfe.support_.tolist()
        
        # copy DataFrame to new temp DataFrame
        copyd3_Ndatatr = d3_Ndatatr.copy()
        col = copyd3_Ndatatr.columns
        
        # drop all non important features
        for sel in range(len(selector)):
            if not selector[sel]:
                copyd3_Ndatatr = copyd3_Ndatatr.drop(col[sel],axis = 1)
        
        # Running Cross Validation
        scores = cross_val_score(LRmodellp,copyd3_Ndatatr,d_targettr,cv=5,n_jobs=-1)   
        bestlp.append([l,p,scores.mean()])

BestLP = pd.DataFrame(bestlp,columns=['l','p','a'])
BestLP.loc[BestLP['a']==max(BestLP['a'])].head(1)



class LogisticReg:
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        #### Get p-values for the fitted model ####
        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
        
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij
        self.coef_ = self.model.coef_[0]
        return self.coef_,self.sigma_estimates,self.z_scores,self.p_values
        
    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y):
        return self.model.score(X,y)

# Create Dataframe of l = 1
d4_datatr = createD(trainlist,1)
d4_Ndatatr = NSE(d4_datatr,createD(dflist,1))
d4_datate = createD(testlist,1)
d4_Ndatate = NSE(d4_datate,createD(dflist,1))

# LR model
LRmodeld4 = LogisticRegression(C = 1e30,n_jobs=-1)

# feature selection of size 7
rfe = RFE(LRmodeld4, 7)
rfe = rfe.fit(d4_Ndatatr,d_targettr)
selector = rfe.support_.tolist()
coef = rfe.estimator_.coef_

# drop all non important features
copyd4_Ndatatr = d4_Ndatatr.copy()
col = copyd4_Ndatatr.columns
for sel in range(len(selector)):
    if not selector[sel]:
        copyd4_Ndatatr = copyd4_Ndatatr.drop(col[sel],axis = 1)
copyd4_Ndatate = d4_Ndatate.copy()
col = copyd4_Ndatate.columns
for sel in range(len(selector)):
    if not selector[sel]:
        copyd4_Ndatate = copyd4_Ndatate.drop(col[sel],axis = 1)

# Lr model fit and predict
LRmodelpv = LogisticReg(C = 1e30,n_jobs=-1)
coefs, se, z, p = LRmodelpv.fit(copyd4_Ndatatr,d_targettr)
prediction = LRmodelpv.predict(copyd4_Ndatate)
confM = confusion_matrix(d_targette,prediction)

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confM, index = ["Other","Bending"],columns = ["Other","Bending"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")

# print table
t = PrettyTable(['Parameter','Coefficient','Standard Error','Z Score','P value'])
for x in range(len(copyd4_Ndatatr.columns)):
    t.add_row([copyd4_Ndatatr.columns[x],round(coefs[x],4),round(se[x],3),round(z[x],3),round(p[x],3)])
print(t)

# plot ROC Curve
fpr, tpr, threshold = roc_curve(d_targette, prediction)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.03, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print("Test Accuracy is",LRmodelpv.score(copyd4_Ndatate,d_targette)*100,"%")
print("Train Accuracy is",max(BestLP['a'])*100,"%")


d7_datatr = createD(trainlist,1)
d7_Ndatatr = NSE(d7_datatr,createD(dflist,1))
d7_datate = createD(testlist,1)
d7_Ndatate = NSE(d7_datate,createD(dflist,1))

# Passing Balanced class weight as paramters which does over sampling implicitly to combact inbalanced classes.
LRmodel = LogisticRegression(C = 1e30,n_jobs=-1,class_weight='balanced')
LRmodel.fit(d7_Ndatatr,d_targettr)

# prediction
prediction = LRmodel.predict(d7_Ndatate)
confM = confusion_matrix(d_targette,prediction)

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confM, index = ["Other","Bending"],columns = ["Other","Bending"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")

# ROC AUC CURVE
fpr, tpr, threshold = roc_curve(d_targette, prediction)
roc_auc = auc(fpr, tpr)

# plotting
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.03, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



bestlc = []
for l in range(1,21):
    # creating DataFrame of training data
    e1_datatr = createD(trainlist,l)
    
    # Normalizing Data
    e1_Ndatatr = NSE(e1_datatr,createD(dflist,l))
    
    # For every c value
    c = 0.00001
    while c<10000:
        
        # Logistic Regression Model Created
        LRmodele = LogisticRegression(C = c, n_jobs=-1,penalty='l1', solver='liblinear')
        LRmodele.fit(e1_Ndatatr,d_targettr)
        
        # Running Cross Validation
        scores = cross_val_score(LRmodele,e1_Ndatatr,d_targettr,cv=5,n_jobs=-1)   
        bestlc.append([l,c,scores.mean()])
        
        c = c * 10
        
BestLC = pd.DataFrame(bestlc,columns=['l','c','a'])
BestLC.loc[BestLC['a']==max(BestLC['a'])].head(1)


e2_datatr = createD(trainlist,1)
e2_datate = createD(testlist,1)
    
# Normalizing Data
e2_Ndatatr = NSE(e2_datatr,createD(dflist,1))
e2_Ndatate = NSE(e2_datate,createD(dflist,1))

# Logistic Regression Model Created
LRmodele2 = LogisticRegression(C = float(BestLC.loc[BestLC['a']==max(BestLC['a'])].head(1)['c']),n_jobs=-1,penalty='l1', solver='liblinear')
result = LRmodele2.fit(e2_Ndatatr,d_targettr)



print("Non Regularized Paramaters")
print("Test Accuracy is",LRmodelpv.score(copyd4_Ndatate,d_targette)*100,"%")
print("Train Accuracy is",max(BestLP['a'])*100,"%")
print("\nRegularized Paramaters")
print("Test Accuracy is",LRmodele2.score(e2_Ndatate,d_targette)*100,"%")
print("Train Accuracy is",max(BestLC['a'])*100,"%")


# training target variables
i = 0
f_targettr = []
for name in range(len(activity_name)):
    for inst in range(train_instance[name]):
        f_targettr.append(activity_name[name])

# testing target variables
i = 0
f_targette = []
for name in range(len(activity_name)):
    for inst in range(test_instance[name]):
        f_targette.append(activity_name[name])



bestmlc = []
for l in range(1,21):
    # creating DataFrame of training data
    f1_datatr = createD(trainlist,l)
    f1_datate = createD(testlist,l)
    # Normalizing Data
    f1_Ndatatr = NSE(f1_datatr,createD(dflist,l))
    f1_Ndatate = NSE(f1_datate,createD(dflist,l))
    
    # For every p value
    c = 0.00001
    while c<10000:
        
        # Logistic Regression Model Created
        LRmodelf = LogisticRegression(C = c, n_jobs=-1,penalty='l1', solver='saga',multi_class='multinomial')
        LRmodelf.fit(f1_Ndatatr,f_targettr)
        
        # Running Cross Validation
        scores = cross_val_score(LRmodelf,f1_Ndatatr,f_targettr,cv=5,n_jobs=-1)   
        bestmlc.append([l,c,scores.mean()])
        
        c = c * 10

BestMLC = pd.DataFrame(bestmlc,columns=['l','c','a'])
BestMLC.loc[BestMLC['a']==max(BestMLC['a'])].head(1)


l = int(BestMLC.loc[BestMLC['a']==max(BestMLC['a'])].head(1)['l'])
c = float(BestMLC.loc[BestMLC['a']==max(BestMLC['a'])].head(1)['c'])

f1_datatr = createD(trainlist,l)
f1_datate = createD(testlist,l)
# Normalizing Data
f1_Ndatatr = NSE(f1_datatr,createD(dflist,l))
f1_Ndatate = NSE(f1_datate,createD(dflist,l))

# Logistic Regression Model Created
LRmodelf1 = LogisticRegression(C = c, n_jobs=-1,penalty='l1',  solver='saga',multi_class='multinomial')
LRmodelf1.fit(f1_Ndatatr,f_targettr)

prediction = LRmodelf1.predict(f1_Ndatate)
pred = LRmodelf1.decision_function(f1_Ndatate)
confM = confusion_matrix(f_targette,prediction)

print("The Test Error is",1 - LRmodelf1.score(f1_Ndatate,f_targette))

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confM, index = activity_name,columns = activity_name)
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")


# plotting
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(activity_name)):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(f_targette, classes=activity_name)[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure()
for i in range(len(activity_name)):
    plt.plot(fpr[i], tpr[i], label='ROC curve '+activity_name[i]+' (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.03, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


f_result = []
for l in range(1,21):
    f2_datatr = createD(trainlist,l)
    f2_datate = createD(testlist,l)
    
    # Gaussian Naive Bayes'
    gnb = GaussianNB()
    # Multinomial Naive Bayes'
    mnb = MultinomialNB()
    
    # fitting data
    gnb.fit(f2_datatr,f_targettr)
    mnb.fit(f2_datatr,f_targettr)
    
    # calculating scores
    score_gnb = cross_val_score(gnb,f2_datatr,f_targettr, cv = 5)
    score_mnb = cross_val_score(mnb,f2_datatr,f_targettr, cv = 5)
    f_result.append([l,score_gnb.mean(),score_mnb.mean()])

f_resultD = pd.DataFrame(f_result,columns=['l','GaussianNB Score','MultinomialNB Score'])
print("The Best L for GNB is",int(f_resultD.loc[f_resultD['GaussianNB Score']==max(f_resultD['GaussianNB Score'])]['l']))
print("The Best L for MNB is",int(f_resultD.loc[f_resultD['MultinomialNB Score']==max(f_resultD['MultinomialNB Score'])]['l']))
f_resultD.head(5)


f2_datatr2 = createD(trainlist,2)
f2_datate2 = createD(testlist,2)
f2_datatr1 = createD(trainlist,1)
f2_datate1 = createD(testlist,1)

# Gaussian Naive Bayes
gnb = GaussianNB()
# Multinomial Naive Bayes'
mnb = MultinomialNB()

# fitting data
gnb.fit(f2_datatr2,f_targettr)
mnb.fit(f2_datatr1,f_targettr)

# predictions
predictiongnb = gnb.predict(f2_datate2)
predictionmnb = mnb.predict(f2_datate1)

predgnb = gnb.predict_proba(f2_datate2)
predmnb = mnb.predict_proba(f2_datate1)
confMgnb = confusion_matrix(f_targette,predictiongnb)
confMmnb = confusion_matrix(f_targette,predictionmnb)

print("The Test Error is",1 - gnb.score(f2_datate2,f_targette))

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confMgnb, index = activity_name,columns = activity_name)
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")


print("The Test Error is",1 - mnb.score(f2_datate1,f_targette))

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confMmnb, index = activity_name,columns = activity_name)
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")


# plotting
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(activity_name)):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(f_targette, classes=activity_name)[:, i], predgnb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure()
for i in range(len(activity_name)):
    plt.plot(fpr[i], tpr[i], label='ROC curve '+activity_name[i]+' (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.03, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# plotting
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(activity_name)):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(f_targette, classes=activity_name)[:, i], predmnb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure()
for i in range(len(activity_name)):
    plt.plot(fpr[i], tpr[i], label='ROC curve '+activity_name[i]+' (area = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.03, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


t = PrettyTable(["Method","Train Score","Test Score"])
t.add_row(["Multinomial Regression",LRmodelf1.score(f1_Ndatatr,f_targettr)*100,LRmodelf1.score(f1_Ndatate,f_targette)*100])
t.add_row(['GaussianNB',gnb.score(f2_datatr2,f_targettr)*100,gnb.score(f2_datate2,f_targette)*100])
t.add_row(['MultinomialNB',mnb.score(f2_datatr1,f_targettr)*100,mnb.score(f2_datate1,f_targette)*100])
print(t)

