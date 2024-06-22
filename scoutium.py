# TALENT HUNT CLASSIFICATION WITH MACHINE LEARNING

# İŞ PROBLEMİ
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.

# VERİ SETİ HİKAYESİ
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri
# futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

# scoutium_attributes.csv
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id’si
# 1: Kaleci - 2: Stoper - 3: Sağ bek - 4: Sol bek - 5: Defansif orta saha - 6: Merkez orta saha - 7: Sağ kanat
# 8: Sol kanat - 9: Ofansif orta saha - 10: Forvet
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import optuna
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Veri Seti Okutma
attribute_df = pd.read_csv("scoutium_attributes.csv", sep=";")
potential_df = pd.read_csv("scoutium_potential_labels.csv", sep=";")

# İki veri setini de merge edelim.
df = pd.merge(attribute_df, potential_df, how="left", on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.head()

# Kaleci sınıfını veri setinden kaldıralım.
df = df.loc[df["position_id"] != 1]

# potential_label içerisindeki below_average sınıfını veri setinden kaldıralım.Çünkü veri setinde %1'lik kısmı oluşturuyor.
df = df.loc[df["potential_label"] != "below_average"]

# Veri seti ile her satırda bir oyuncu olacak şekilde yeni bir dataframe oluşturalım.
df_pivot = pd.pivot_table(df, index=["player_id", "position_id", "potential_label"],
                          columns="attribute_id", values="attribute_value")
df_pivot.head()

# Index hatasından ve sütunları str çeviriyorum.
df_pivot = df_pivot.reset_index(drop=False)
df_pivot.columns = [str(col) for col in df_pivot.columns]

# Keşifçi Veri Analizi
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include="number").quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df_pivot)

# Değişkenleri Sınıflandıralım.

# Numerik Değişkenleri Seçelim.
num_cols = df_pivot.select_dtypes(include=["float64"]).columns
# Kategorik Değişkenleri Seçelim.
cat_cols = [col for col in df_pivot.columns if col not in num_cols and col not in "player_id"]

# Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyelim.
# Kategorik Değişkenler için:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df_pivot, col)

# Numerik değişkenler için:
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    print(f"############## {col} ############")
    num_summary(df_pivot, col)

#  Numerik değişkenler ile hedef değişken incelemesini yapalım.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df_pivot, "potential_label", col)

#  Kategorik değişkenler ile hedef değişken incelemesini yapalım.

# İlk olarak hedef değişkenimiz olan "potential_label" 1-0 formatına çevirelim.
df_pivot["potential_label"] = np.where(df_pivot["potential_label"] == "highlighted", 1, 0)
def target_category(dataframe,  target, col_category):
    print(dataframe.groupby(col_category).agg({target: "mean"}))
    print("#" * 40)

for col in cat_cols:
    print(f"######### {col.upper()} #########")
    target_category(df_pivot, "potential_label", col)

# Aykırı Değer İncelemesi Yapalım.
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe.loc[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df_pivot, col))  # AYKIRI DEĞER YOKTUR.

# Eksik Gözlem Var mı İnceleyelim.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns

missing_values_table(df_pivot)  # EKSİK DEĞER YOK.

# Korelasyon İnceleyelim.
def corr_map(df, width=14, height=6, annot_kws=15, corr_th=0.7):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize = (width,height))
    sns.heatmap(df.corr(),
                annot= True,
                fmt = ".2f",
                ax=ax,
                vmin = -1,
                vmax = 1,
                cmap = "RdBu",
                mask = mtx,
                linewidth = 0.4,
                linecolor = "black",
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0,size=15)
    plt.xticks(rotation=75,size=15)
    plt.title('\nCorrelation Map\n', size = 40)
    plt.show()
    return drop_list
corr_map(df_pivot[num_cols])

# BASE MODEL
df_base = df_pivot.copy()
# Encoding İşlemleri
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = ["position_id"]
for col in binary_cols:
    label_encoder(df_base, col)

# Scale İşlemleri
rs_scale = RobustScaler()
df_base[num_cols] = rs_scale.fit_transform(df_base[num_cols])

# Model Aşaması
X = df_base.drop(["potential_label", "player_id"], axis=1)
y = df_base["potential_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

models = [("LR", LogisticRegression()),
          ("CART", DecisionTreeClassifier()),
          ("KNN", KNeighborsClassifier()),
          ("GBM", GradientBoostingClassifier()),
          ("RF", RandomForestClassifier()),
          ("XGBoost", XGBClassifier()),
          ("LightGBM", LGBMClassifier(verbosity=-1)),
          ("CatBoost", CatBoostClassifier(verbose=False))]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"########### {name} ###########")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 3)}")
    print(f"Precision: {round(precision_score(y_test, y_pred), 3)}")
    print(f"Recall: {round(recall_score(y_test, y_pred), 3)}")
    print(f"F1 Score: {round(f1_score(y_test, y_pred), 3)}")
    print(f"Roc_Auc: {round(roc_auc_score(y_test, y_prob), 3)}")

########### LR ###########
# Accuracy: 0.8
# Precision: 0.667
# Recall: 0.167
# F1 Score: 0.267
# Roc_Auc: 0.727

########### CART ###########
# Accuracy: 0.818
# Precision: 0.625
# Recall: 0.417
# F1 Score: 0.5
# Roc_Auc: 0.673

########### KNN ###########
# Accuracy: 0.782
# Precision: 0.5
# Recall: 0.083
# F1 Score: 0.143
# Roc_Auc: 0.473

########### GBM ###########
# Accuracy: 0.818
# Precision: 0.625
# Recall: 0.417
# F1 Score: 0.5
# Roc_Auc: 0.793

########### RF ###########
# Accuracy: 0.818
# Precision: 1.0
# Recall: 0.167
# F1 Score: 0.286
# Roc_Auc: 0.918

########### XGBoost ###########
# Accuracy: 0.836
# Precision: 0.714
# Recall: 0.417
# F1 Score: 0.526
# Roc_Auc: 0.775

########### LightGBM ###########
# Accuracy: 0.764
# Precision: 0.429
# Recall: 0.25
# F1 Score: 0.316
# Roc_Auc: 0.762

########### CatBoost ###########
# Accuracy: 0.8
# Precision: 0.6
# Recall: 0.25
# F1 Score: 0.353
# Roc_Auc: 0.872

# Veri setinde dengesizlik olduğu için F1 skoruna göre model olarak en başarılı XGBoost belirliyorum.

# FEATURE ENGINEERING
# Pozisyonlara göre defans mı atak mı?
df_pivot["position_mental"] = df_pivot["position_id"].apply(lambda x: "defender" if (x == 2) or (x == 3) or (x == 4)
                                                                                    or (x == 5) else "attacker")
# Verilen puanların içinden sayısal değerler
df_pivot["min"] = df_pivot[num_cols].min(axis=1)
df_pivot["max"] = df_pivot[num_cols].max(axis=1)
df_pivot["mean"] = df_pivot[num_cols].mean(axis=1)
df_pivot["sum"] = df_pivot[num_cols].sum(axis=1)
df_pivot["median"] = df_pivot[num_cols].median(axis=1)
df_pivot["std"] = df_pivot[num_cols].std(axis=1)

# Her oyuncuya verilen puanın genele karşı durumu
for i in df_pivot.columns[3:-7]:  # for i in num_cols:
    threshold = df_pivot[i].mean() + df_pivot[i].std()
    df_pivot[str(i) + "_FLAG"] = df_pivot[i].apply(lambda x: 0 if x < threshold else 1)

# Sporcuların başarı durumları
flagCols = [col for col in df_pivot.columns if "_FLAG" in col]
df_pivot["counts"] = df_pivot[flagCols].sum(axis=1)
df_pivot["countRatio"] = df_pivot["counts"] / len(flagCols)

# MACHINE LEARNING MODEL

# Encoding İşlemleri
# Sadece position_mental değişkenini encode edeceğim.
binary_cols = ["position_mental"]
for col in binary_cols:
    label_encoder(df_pivot, col)

# Scale İşlemleri
scale_cols = [col for col in df_pivot.columns if df_pivot[col].dtypes in ["int64", "float64"] and df_pivot[col].nunique() > 2]
scale_cols = [col for col in scale_cols if col not in ["player_id", "position_id"]]

rs_scale = RobustScaler()
df_pivot[scale_cols] = rs_scale.fit_transform(df_pivot[scale_cols])

# Model Aşaması
X = df_pivot.drop(["player_id", "potential_label"], axis=1)
y = df_pivot["potential_label"]

xg_model = XGBClassifier()
cv_results = cross_validate(xg_model, X, y, cv=5, scoring=["accuracy", "f1", "recall", "precision", "roc_auc"])

print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Accuracy: 0.8562
# Auc: 0.8684
# Recall: 0.55
# Precision: 0.7227
# F1: 0.6015

# Hiperparametre Optimizasyonu yapalım.
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [3, 5, 8],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.7, 1]}

xg_model_best_grid = GridSearchCV(xg_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# En iyi hiperparametreler
# {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 300}

xg_model_final = xg_model.set_params(**xg_model_best_grid.best_params_).fit(X, y)
cv_results_final = cross_validate(xg_model_final, X, y, cv=5, scoring=["accuracy", "f1", "recall", "precision", "roc_auc"])

print(f"Accuracy: {round(cv_results_final['test_accuracy'].mean(), 4)}")
print(f"Auc: {round(cv_results_final['test_roc_auc'].mean(), 4)}")
print(f"Recall: {round(cv_results_final['test_recall'].mean(), 4)}")
print(f"Precision: {round(cv_results_final['test_precision'].mean(), 4)}")
print(f"F1: {round(cv_results_final['test_f1'].mean(), 4)}")

# Accuracy: 0.8747
# Auc: 0.8728
# Recall: 0.5864
# Precision: 0.7979
# F1: 0.652

# Feature Engineering ve Hiperparametre optimizasyonu ile F1 skorumuzu 0.52'den 0.65'e çıkardık.

# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdirelim.
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(xg_model_final, X, num=15)
# Oluşturduğumuz matematiksel değikenlerin modele etki ettiğini görüyoruz.

# Modelimize random bir player soralım.
random_user = X.sample(1, axis=0)
xg_model_final.predict(random_user)  # 1 tahmini yani highlighted bir oyuncu








