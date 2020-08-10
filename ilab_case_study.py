# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:12:43 2020

@author: suleyman.kaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats

#Kullanılacak Ham Verininin Yüklenmesi
all_data = pd.read_csv("term-deposit-marketing-2020.csv")

#Ham Verinin Analizi
all_data.info()
print("-----------------------------------------------------------------")
all_data_types= all_data.dtypes
all_data_details= all_data.describe()
all_data_details2= all_data.describe(include="all")


#Eksik Verilerin En Sık Bulunan Veriler İle Değiştirilmesi
all_data.replace("unknown", np.nan, inplace= True)
all_data = all_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
#Most_Frequent_Data = all_data['job'].value_counts().argmax()

"""
Eksik Verilerin En Sık Bulunan Veriler İle Değiştirilmeden Önceki Sayıları
Eksik (unknown) Veri Sayısı - job: 235 | education: 1531 | contact:12765
"""
#Eksik Verilerin Olup Olmadığının Tespiti (Varsa "True" olarak veri setinde gösterir)
missing_data = all_data.isnull()

#Her Alan İçin Eksik Verilerin Gösterimi
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

print("-----------------------------------------------------------------")

#Kopyalama İşlemi
all_data2 = all_data

#Encoder: Kategorik Verilerin Sayısallıştırılması
all_data2 = all_data2.apply(LabelEncoder().fit_transform)

#all_data2 = all_data3 + marital_contact_data 
marital_contact_data = all_data2[["marital", "contact"]]
all_data3 = all_data2.drop(["marital", "contact"], 1)

#Kopyalama İşlemi
all_data4 = marital_contact_data 

#OneHotEncoder İşlemi
ohe = OneHotEncoder(categorical_features='all')
all_data4 = ohe.fit_transform(all_data4).toarray()

#Alanların İsimlendirilmesi İşlemi
all_data4 = pd.DataFrame(data = all_data4, index = range(40000), columns=['divorced','married','single', 'telephone','cellular'] )

#Hazırlanan Verilen Birleştirme İşlemi
all_data5 = pd.concat([all_data3, all_data4], axis=1)


#Outliers Verilerin Sistemden Çıkartılması İşlemi
for column in all_data5:
    all_data5['z_score']=stats.zscore(all_data5[column])
    all_data5.rename(columns={'z_score': 'z_score of '+column}, inplace=True)
    all_data5 = all_data5[all_data5['z_score of '+column].abs()<=5]
    all_data5 = all_data5.drop('z_score of '+column, 1)

#Bağımlı ve Bağımsız Değişkenlerin Oluşturulması
x_input = all_data5.drop('y',1)
y_output = all_data5['y']

#Olasılık Değeri (p-value) ve Korelasyon Değerine Bağlı Olarak Geri Eleme (Backward Elimination) Yapma    
#p_and_corr Verisi Müşterilerin Ürünü En Çok Almasında Etkili Özellikler Kod Bloğunda Kullanılacaktır
unrelated_data = []
p_and_corr = pd.DataFrame(columns=["p_value","pearson_coef"])

for column in x_input:
    pearson_coef, p_value = stats.pearsonr(x_input[column], y_output)
    print('Alan Adı: ', column)
    print("p_value: ", p_value, "|", "pearson_coef: ", pearson_coef)
    p_and_corr = p_and_corr.append({'p_value':p_value , 'pearson_coef':pearson_coef}, ignore_index=True)
    unrelated_data_check = p_value<0.05 and abs(pearson_coef)>0.01
    if unrelated_data_check == False:
        unrelated_data.append(column)

p_and_corr.index = x_input.columns

#İlişkisi Çok Az Olan Alanların Veri Setinden Çıkarılması İşlemi
x_input = x_input.drop(unrelated_data,1)
 
print("-----------------------------------------------------------------")


#Veri Setinin Eğitim ve Test Kümelerine Bölünmesi
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size = 0.33, random_state = 0)

#Verilerin Ölçeklendirilmesi
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#SVM ile Öğrenme
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#SVM İle Tahmin
y_pred = classifier.predict(x_test)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("-----------------------------------------------------------------")
print(cm)
print('Accuracy Score:', accuracy_score(y_test, y_pred)*100)
print("-----------------------------------------------------------------")
#5-Fold Cross Validation
success = cross_val_score(estimator = classifier, X=x_train, y=y_train , cv = 5)
print("-----------------------------------------------------------------")
print("ML Model Success Rate: ", success.mean()*100)
print("-----------------------------------------------------------------")
#print(success.std())

"""
Note1:
Case Study Success Metric : Hit %81 or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.
Result Success Metrics: Accuracy Score and ML Model Success Rate are "%93".
"""


#Abone Olma Olasılığı En Yüksek Müşteriler (Büyüktün Küçüğe Doğru) = interested_customers
dvr_reg= SVR(kernel="rbf")
dvr_reg.fit(x_train, y_train)
y_pred2 = dvr_reg.predict(x_test)
interested_customers = pd.DataFrame(data = y_pred2, columns=['Potantial_Customers']).sort_values('Potantial_Customers', ascending=False)

#Müşterilerin Ürünü Satın Almasında En Çok Etkili Olan Özellikler
#Sırasıyla En Çok Etkileyen 3 Parametre: 1.DURATION | 2.MARRIED | 3.HOUSING
p_and_corr = p_and_corr.abs()
p_values_data = p_and_corr.iloc[:,0].sort_values(ascending=True)
print("Related Faetures1:\n", p_values_data.head(3))
corr_data = p_and_corr.iloc[:,1].sort_values(ascending=False)
print("Related Faetures2:\n", corr_data.head(3))
print("-----------------------------------------------------------------")

#Gerçek ve Tahmin Edilen Verilerin Grafiksel Olarak Karşılaştırılması
y_pred = pd.Series(y_pred).rename('y')
y_test.plot(kind='hist', x='index', y='y', color='red', figsize=(12,8))
plt.show()
y_pred.plot(kind='hist', x='index', y='y', color='blue', figsize=(12,8))
plt.show()

"""
Note2:
Bonuses:    
- We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? Tell us which feature we should be focusing more on.
"""

