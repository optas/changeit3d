import pandas as pd
from ..basics import splitall


def unary_split_based_on_shape_split(shape_split_file, df):
    split_df = pd.read_csv(shape_split_file)

    # make a dictionary from <shape-stimulus> to <split>
    uid = split_df.shape_class + '/' + split_df.dataset + '/' + split_df.model_name
    split_df = split_df.assign(uid=uid)
    uids_to_split = split_df.groupby(uid)['split'].apply(lambda x: list(x)[0]).to_dict()

    df = df.assign(target_unary_split=df.target_uid.apply(lambda x: uids_to_split[x]))
    df = df.assign(source_unary_split=df.source_uid.apply(lambda x: uids_to_split[x]))
    return df


def decide_listening_split_based_on_shape_split(df, rule='respect_target_only', verbose=False):
    if rule == 'respect_target_only':
        result = df.target_unary_split

    if verbose:
        for split in result.unique():
            print(split, (result == split).mean())

    return result


def file_name_to_meta_data(file_name, ending='.png'):
    tokens = splitall(file_name)
    model_name = tokens[-1][:-len(ending)]
    dataset = tokens[-2]
    return model_name, dataset



