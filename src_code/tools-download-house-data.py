import hashlib
import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# 下载房屋信息压缩包
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data(HOUSING_URL,HOUSING_PATH)

# 加载房屋信息
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# 分割出训练数据
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()

# 把索引设置为id，分出训练集和测试集
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# 把经纬度设置为id，分出训练集和测试集
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# 用sklearn的train_test_split方法分离
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# 用sklearn的StratifiedShuffleSplit方法分离
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 创造一些对机器学习 真正有意义的数据
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# 计算每一个属性和房价中位数的相关度
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# DATA CLEANING
# housing.dropna(subset=["total_bedrooms"]) #第一种方式 去掉没有值得部分
# housing.drop("total_bedrooms",axis=1) #第二种方式 去掉整行数目

median = housing["total_bedrooms"].median()  # 获取中位数
housing["total_bedrooms"].fillna(median)  # 第三种方式，没有值的地方，赋值给中位数

# 使用Imputer处理空值数据
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
# 处理非数字属性数据 imputer不能处理非数字属性
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# LableEncoder
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# # print(encoder.classes_)

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot.toarray())

# LabelBinarizer
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer(sparse_output=True)  # 输出稀疏矩阵格式
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)

# Custom Transformation 自定义转换类
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, bedrooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room == False)
        housing_extra_attribs = attr_adder.transform(housing.values)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('sd_scaler', StandardScaler()), ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('sd_scaler', StandardScaler())
                         ])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('label_binarizer', LabelBinarizer())
                         ])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline)
                                               ])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# SELECT AND TRAIN MODEL 选择训练一个模型
# 1、线性回归模型
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("预测结果:\t", lin_reg.predict(some_data_prepared))
# print("标签:\t\t", some_labels)

# 衡量模型
from sklearn.metrics import mean_squared_error

# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print("衡量结果：", lin_rmse)

# 2、决策树模型
from sklearn.tree import DecisionTreeRegressor

# 训练模型
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)

# evaluate it on the training set
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)


# K-fold-cross-validation
def display_scores(scores):
    print("scores:", scores)
    print("平均数:", scores.mean())
    print("标准差 Standard deviation:", scores.std())


from sklearn.model_selection import cross_val_score

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# display_scores(rmse_scores)

# compute the same scores for the Linear Regression model just to be sure
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)

# 随机森林决策
from sklearn.ensemble import RandomForestRegressor

# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
#
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

# cvres = grid_search.cv_results_
# for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
#     print(np.sqrt(-mean_score),params)

# 打印特征重要性排序
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
# display these importance scores next their corresponding attribute names
extra_attribs = ["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs+cat_one_hot_attribs
print(sorted(zip(feature_importances,attributes),reverse=True))

# print(housing.info())
# print("*****" * 20)
# print(housing["ocean_proximity"].value_counts())
# print("*****" * 20)
# print(housing.describe())
# 柱状图
# housing.hist(bins=50, figsize=(20, 15))

# 计算每一个属性和房价中位数的相关度
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 点图显示价格分布和坐标相关度
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,s=housing["population"]/100,label="population",c="median_house_value",cmap=plt.get_cmap("jet"),
#              colorbar=True)
# plt.legend()

# 显示这四个元素的相关图谱
from pandas.tools.plotting import scatter_matrix

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()
