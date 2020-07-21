import pandas as pd
from sklearn.preprocessing import LabelEncoder

from generate_feature import (built_year, walk_time, area1, area2, maguchi,
                              remarks, landshape, structure, use, num_of_rooms,
                              ldk, total_floor_area_div_area,
                              total_floor_area_per_floor,
                              area_div_frontage, frontage_div_breadth)


def category_encode(df, target_cols):
    for col in target_cols:
        df[col] = df[col].fillna("NaN")
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df


def preprocess(df):
    df = built_year(df)
    df = walk_time(df)
    df = area1(df)
    df = area2(df)
    df = maguchi(df)
    df = ldk(df)
    df = num_of_rooms(df)
    df = total_floor_area_div_area(df)
    df = total_floor_area_per_floor(df)
    df = area_div_frontage(df)
    df = frontage_div_breadth(df)
    # df = remarks(df)
    # df = landshape(df)
    # df = structure(df)
    # df = use(df)
    return df
