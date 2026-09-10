import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from scipy.stats import skew

warnings.filterwarnings('ignore')

# 1. 数据加载与基础处理
train_df, test_df = pd.read_csv('train.csv'), pd.read_csv('test.csv')
train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index, inplace=True)

test_ID = test_df['Id']
train_df.drop("Id", axis=1, inplace=True)
test_df.drop("Id", axis=1, inplace=True)

rank_dict = {n: r for r, n in enumerate(train_df.groupby('Neighborhood')['SalePrice'].median().sort_values().index, 1)}
y_train = np.log1p(train_df.pop("SalePrice"))
ntrain = train_df.shape[0]

all_data = pd.concat([train_df, test_df]).reset_index(drop=True)
all_data['Neighborhood'] = all_data['Neighborhood'].map(rank_dict)
all_data['Neighborhood'] = all_data['Neighborhood'].fillna(all_data['Neighborhood'].median())

for c in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']: all_data[c] = all_data[c].astype(str)

none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
all_data[none_cols] = all_data[none_cols].fillna('None')
all_data[zero_cols] = all_data[zero_cols].fillna(0)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
for c in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    all_data[c] = all_data[c].fillna(all_data[c].mode()[0])

# 2. 特征工程与偏态平滑
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath']
all_data['TotalPorch'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
all_data['HouseAge'] = all_data['YrSold'].astype(int) - all_data['YearBuilt']
all_data['YearsSinceRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd']
all_data['IsRemodeled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)


num_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed = all_data[num_feats].apply(lambda x: skew(x.dropna()))
all_data[skewed[abs(skewed) > 0.75].index] = np.log1p(all_data[skewed[abs(skewed) > 0.75].index])

# 3. 编码、插补与 PCA
ord_cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir']
for c in [c for c in ord_cols if c in all_data.columns]:
    all_data[c] = LabelEncoder().fit_transform(all_data[c].astype(str))

all_data = pd.get_dummies(all_data)

scaler = RobustScaler()
all_data = pd.DataFrame(scaler.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(scaler.fit_transform(all_data))), columns=all_data.columns)

X_train, X_test = all_data[:ntrain], all_data[ntrain:]

data_pca = all_data.copy()
data_pca['Gar_PCA'] = PCA(n_components=1).fit_transform(RobustScaler().fit_transform(data_pca[['GarageCars', 'GarageArea']]))
data_pca['Bsmt_PCA'] = PCA(n_components=1).fit_transform(RobustScaler().fit_transform(data_pca[['TotalBsmtSF', '1stFlrSF']]))
data_pca.drop(['GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF'], axis=1, inplace=True)
X_train_pca, X_test_pca = data_pca[:ntrain], data_pca[ntrain:]

# 4. 自动评估与预测生成
models = {
    'PCA+OLS': (LinearRegression(), X_train_pca, X_test_pca),
    'RidgeCV': (RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0], cv=5), X_train, X_test),
    'LassoCV': (LassoCV(alphas=[0.0001, 0.0003, 0.0005, 0.001, 0.005], cv=5, random_state=42), X_train, X_test),
    'ElasticNetCV': (ElasticNetCV(alphas=[0.0001, 0.0003, 0.0005, 0.001], l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42), X_train, X_test)
}

best_rmse, best_preds, best_name = float('inf'), None, ""

print("各方法本地交叉验证 RMSE:")
for name, (model, X_tr, X_te) in models.items():
    rmse = -cross_val_score(model, X_tr, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
    print(f" - {name:<15}: {rmse:.5f}")
    if rmse < best_rmse:
        best_rmse, best_preds, best_name = rmse, model.fit(X_tr, y_train).predict(X_te), name

print(f"\n最优模型 [{best_name}] 已选中，正在生成 submission.csv...")
pd.DataFrame({'Id': test_ID, 'SalePrice': np.expm1(best_preds)}).to_csv('submission.csv', index=False)