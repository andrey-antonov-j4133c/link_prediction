import pandas as pd


def data_split(df, features_df):
    df0 = df[df['goal'] == 0]
    df1 = df[df['goal'] == 1]

    l = int(len(df0) * 0.333)

    link_prediction_train = pd.concat([df0.sample(l), df1.sample(l)])

    remainder = pd.concat([df, link_prediction_train, link_prediction_train]).drop_duplicates(keep=False)

    link_prediction_test = pd.concat([
        remainder[remainder['goal'] == 0].sample(l),
        remainder[remainder['goal'] == 1].sample(l)
    ])

    remainder = pd.concat([df, link_prediction_test, link_prediction_test]).drop_duplicates(keep=False)

    classifier_test = pd.concat([
        remainder[remainder['goal'] == 0].sample(l),
        remainder[remainder['goal'] == 1].sample(l)
    ])

    return [
        link_prediction_train[['node1', 'node2', 'goal']].merge(features_df, how='inner', on=['node1', 'node2']),
        link_prediction_test[['node1', 'node2', 'goal']].merge(features_df, how='inner', on=['node1', 'node2']),
        classifier_test[['node1', 'node2', 'goal']].merge(features_df, how='inner', on=['node1', 'node2'])
    ]
