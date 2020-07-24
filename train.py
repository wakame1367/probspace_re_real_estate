from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nyaggle.experiment import run_experiment
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from preprocess import preprocess, category_encode

data_path = Path("resources")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def load_dataset():
    train = pd.read_csv(data_path / "train_data.csv")
    test = pd.read_csv(data_path / "test_data.csv")
    rename_pairs = {
        "所在地コード": "市区町村コード", "建蔽率": "建ぺい率（％）",
        "容積率": "容積率（％）", "駅名": "最寄駅：名称",
        "地積": "面積（㎡）", "市区町村名": "市区町村名",
        '前面道路の幅員': '前面道路：幅員（ｍ）', "前面道路の方位区分": "前面道路：方位",
        "前面道路区分": "前面道路：種類", "形状区分": "土地の形状",
        "用途区分": "都市計画", '用途': '地域'
    }
    land_price = pd.read_csv(data_path / "published_land_price.csv",
                             dtype={'利用の現況': str})
    land_price = land_price.rename(columns=rename_pairs)
    return train, test, land_price


def preprocess_land_price(land_price):
    land_price['最寄駅：距離（分）'] = land_price['駅距離'] // 50
    land_price.loc[:, '最寄駅：距離（分）'][land_price['最寄駅：距離（分）'] > 120] = 120
    land_price['間口（比率）'] = land_price['間口（比率）'].clip(10, 100)
    land_price['奥行（比率）'] = land_price['奥行（比率）'].clip(10, 100)
    land_price['間口'] = np.sqrt(
        land_price['面積（㎡）'] / land_price['間口（比率）'] / land_price[
            '奥行（比率）']) * land_price['間口（比率）']

    # 東京府中 -> 府中
    land_price["市区町村名"] = land_price["市区町村名"].replace(r"^東京", "",
                                                      regex=True)
    # train/testと統一
    land_price["市区町村名"] = land_price["市区町村名"].str.replace('ケ', 'ヶ')
    land_price["面積（㎡）"] = land_price["面積（㎡）"].clip(0, 3000)
    # preprocess 利用の現況
    # 最新の公示価格を対象
    target_col = "Ｈ３１価格"
    target = land_price[target_col]
    target = target.rename("land_price")
    # train/test 取引金額(1,000,000円)表記に合わせる
    target = target / 100000
    # drop_pat_1 = '(Ｓ|Ｈ).+価格'
    # drop_pat_2 = '属性移動(Ｓ|Ｈ).+'
    # 容積率（％）までのカラムを用いる
    land_price = land_price.iloc[:, :41]
    land_price["取引時点"] = 2019
    land_price = land_price.join(target)

    # land_price = land_price.rename({"緯度": "latitude", "経度": "longitude"})
    rep = {'1低専': '第１種低層住居専用地域',
           '2低専': '第２種低層住居専用地域',
           '1中専': '第１種中高層住居専用地域',
           '2中専': '第２種中高層住居専用地域',
           '1住居': '第１種住居地域',
           '2住居': '第２種住居地域',
           '準住居': '準住居地域', '商業': '商業地域', '近商': '近隣商業地域',
           '工業': '工業地域', '工専': '工業専用地域', '準工': '準工業地域', '田園住': '田園住居地域'}
    for key, value in rep.items():
        land_price.loc[:, '都市計画'] = land_price.loc[:, '都市計画'].str.replace(key,
                                                                          value)
    land_price = land_price.rename(columns={'利用の現況': '用途'})

    # 住所番地手前で切り出し
    se = land_price['住居表示'].str.strip('東京都').str.replace('大字',
                                                         '').str.replace(
        '字', '')
    # 番地総当たり
    for num in ['１', '２', '３', '４', '５', '６', '７', '８', '９']:
        se = se.str.split(num).str[0].str.strip()
    land_price['地区詳細'] = se
    land_price['地区詳細'] = land_price['地区詳細'].str[:5]
    return land_price


def current_status_of_use(land_price):
    riyo_list = np.array(
        ['住宅', '店舗', '事務所', '_', '_',
         '_', '工場', '倉庫', '_', '_', '_', '_',
         '作業場', '_', 'その他', '_', '_']
    )
    riyo_now = [[0] * (17 - len(num)) + list(map(int, list(num)))
                for num in land_price['利用の現況'].values]
    riyo_now = np.array(riyo_now)
    riyo_lists = ['、'.join(riyo_list[onehot.astype('bool')]) for onehot in
                  riyo_now]
    for i in range(len(riyo_lists)):
        if 'その他' in riyo_lists[i]:
            riyo_lists[i] = riyo_lists[i].replace('その他', land_price.loc[
                i, '利用状況表示'])
        riyo_lists[i] = riyo_lists[i].replace('_', 'その他').replace('、雑木林',
                                                                  '').replace(
            '、診療所', '').replace('、車庫', '').replace('、集会場', '') \
            .replace('、寄宿舎', '').replace('、駅舎', '').replace('、劇場', '').replace(
            '、物置', '').replace('、集会場', '').replace('、映画館', '') \
            .replace('、遊技場', '').replace('兼', '、').replace('、建築中',
                                                           'その他').replace(
            '、試写室', '').replace('、寮', '').replace('、保育所', '') \
            .replace('、治療院', '').replace('、診療所', '').replace('、荷捌所',
                                                             '').replace('建築中',
                                                                         'その他').replace(
            '事業所', '事務所').replace('、営業所', '')
    land_price['利用の現況'] = riyo_lists
    return land_price


def clean_land_price(df):
    target_col = "市区町村名"
    # 東京府中 -> 府中
    df[target_col] = df[target_col].replace(r"^東京", "", regex=True)
    return df


def clean_train_test(df):
    target_col = "市区町村名"
    # 西多摩郡日の出 -> 日の出
    df[target_col] = df[target_col].replace(r"^西多摩郡", "", regex=True)
    df[target_col] = df[target_col].map(lambda x: x.rstrip("市区町村"))
    return df


def add_landp(train, test, land_price):
    # 直近5年のみ対象
    target_cols = ["Ｈ２７価格", "Ｈ２８価格", "Ｈ２９価格", "Ｈ３０価格", "Ｈ３１価格"]
    land_price["landp_mean"] = land_price[target_cols].mean(axis=1)
    landp_mean = land_price.groupby("市区町村名")["landp_mean"].mean().reset_index()
    train = train.merge(landp_mean, on='市区町村名')
    test = test.merge(landp_mean, on='市区町村名')
    return train, test


def add_lat_and_long(train, test, land_price):
    lat_and_long = land_price.groupby("市区町村名")[
        "latitude", "longitude"].mean().reset_index()
    train = train.merge(lat_and_long, on='市区町村名')
    test = test.merge(lat_and_long, on='市区町村名')
    return train, test


def main():
    with open("settings/colum_names.yml", "r", encoding="utf-8") as f:
        rename_dict = yaml.load(f, Loader=yaml.Loader)

    train, test, land_price = load_dataset()

    target_col = "y"

    target = train[target_col]
    target = target.map(np.log1p)
    test[target_col] = -1
    # train.drop(columns=[target_col], inplace=True)
    _all = pd.concat([train, test], ignore_index=True)
    _all["地区詳細"] = _all['市区町村名'] + _all['地区名']
    _all["地区詳細"] = _all["地区詳細"].str[:5]
    land_price = preprocess_land_price(land_price)
    # cols = ['地域', '市区町村コード', '地区詳細', '建ぺい率（％）', '容積率（％）', '都市計画', '前面道路：方位',
    #         '前面道路：種類', '最寄駅：名称']
    merge_keys = ['市区町村コード', '地区詳細', '最寄駅：名称']
    land_price_col = "land_price"
    merge_columns = [merge_key + "_" + land_price_col for merge_key in merge_keys]
    for merge_key, rename_col in zip(merge_keys, merge_columns):
        # print(col, _all.shape)
        group_mean = land_price[[merge_key, land_price_col]].groupby(
            merge_key).mean()
        group_mean = group_mean.rename(columns={land_price_col: rename_col})
        _all = pd.merge(_all, group_mean, on=merge_key, how='left')

    a = '地区詳細' + "_" + land_price_col
    b = '市区町村コード' + "_" + land_price_col
    c = '最寄駅：名称' + "_" + land_price_col

    # nanになる値を他カラムの値を用いて埋める
    _all.loc[_all[a].isna(), a] = _all.loc[_all[a].isna(), b]
    _all.loc[_all[c].isna(), c] = _all.loc[_all[c].isna(), a]

    # for merge_key, rename_col in zip(merge_keys, merge_columns):
    #     _all['m2x' + merge_key] = _all[rename_col] * _all['面積（㎡）'] / 100
    #     # _all['nm2x'+col] = _all[col+'y'] * _all['延床面積（㎡）']/100
    #     _all['m2m2x' + merge_key] = _all[rename_col] * (
    #             _all['面積（㎡）'] + _all['延床面積（㎡）'].fillna(0)) / 100
    #     _all['m2m2x' + merge_key + '_sta'] = _all['m2m2x' + merge_key] * (1 - _all[
    #                                                     '最寄駅：距離（分）'].clip(0,
    #                                                                       10) * 0.02)
    _all = _all.rename(columns=rename_dict)
    land_price = land_price.rename(columns=rename_dict)
    _all = preprocess(_all)
    drop_cols = ["id", "Prefecture", "Municipality", "年号", "和暦年数", 'FloorPlan']
    one_hot_cols = ['Structure', 'Use', 'Remarks']
    cat_cols = ['Type', 'Region', 'MunicipalityCode', 'DistrictName',
                'NearestStation', 'LandShape', 'Purpose',
                'Direction', 'Classification', 'CityPlanning', 'Renovation',
                'Period', '地区詳細']
    _all.drop(columns=drop_cols, inplace=True)

    _all = category_encode(_all, cat_cols + one_hot_cols)

    _all = _all.rename(columns={
        '地区詳細': "a",
        '市区町村コード_land_price': "b",
        '地区詳細_land_price': "c",
        '最寄駅：名称_land_price': "d"})

    train = _all[_all[target_col] >= 0]
    test = _all[_all[target_col] < 0]
    # target = _all[_all[target_col] >= 0].loc[:, target_col]

    train.drop(columns=[target_col], inplace=True)
    test.drop(columns=[target_col], inplace=True)
    del _all

    lightgbm_params = {
        "metric": "rmse",
        "objective": 'regression',
        "max_depth": 5,
        "num_leaves": 24,
        "learning_rate": 0.007,
        "n_estimators": 30000,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 1,
        "reg_alpha": 0,
        "reg_lambda": 0,
    }

    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 5000
    }

    n_splits = 4
    kf = KFold(n_splits=n_splits)

    logging_directory = "resources/logs/lightgbm/{time}"
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging_directory = logging_directory.format(time=now)
    lgb_result = run_experiment(lightgbm_params,
                                X_train=train,
                                y=target,
                                X_test=test,
                                eval_func=rmse,
                                cv=kf,
                                fit_params=fit_params,
                                logging_directory=logging_directory)

    # too long name
    submission = lgb_result.submission_df
    submission[target_col] = submission[target_col].map(np.expm1)
    # replace minus values to 0
    _indexes = submission[submission[target_col] < 0].index
    submission.loc[_indexes, target_col] = 0
    # index 0 to 1
    submission["id"] += 1
    sub_path = Path(logging_directory) / "{}.csv".format(now)
    submission.to_csv(sub_path, index=False)


if __name__ == '__main__':
    main()
