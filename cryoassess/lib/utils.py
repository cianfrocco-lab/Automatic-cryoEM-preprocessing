'''
Conversion between dataframes and star files in Relion.
ABOUT star_df:
    star_df is a dictionary:
        Keys: blockcodes ('data_xx');
        Values: lists of pd dataframes:
            Each element in the value/list corresponds to a "data block", start
            with "loop_", and converted to a pd dataframe.
                Each column name of the dataframe is "data name" (e.g. _rlnMicrographName)
                All data in the dataframe is stored as strings.
    The commented part started with "#" are deleted during the conversion.
'''

# import numpy as np
import pandas as pd

def loop2df(loop):
    keys_idx = [i for i, x in enumerate(loop) if x.startswith('_')]
    keys = [loop[i].split('#',1)[0].strip() for i in keys_idx] # remove everything after the first "#" on the keys

    df = loop[keys_idx[-1]+1:]
    df = [x.split() for x in df]
    df = pd.DataFrame(df).dropna()
    df.columns = keys

    return df

def block2df(block):
    loop_idx = [i for i, x in enumerate(block) if x == 'loop_']
    loop_idx.append(len(block))
    loops = [block[loop_idx[i]:loop_idx[i+1]] for i in range(len(loop_idx)-1)]

    df_list = []
    for loop in loops:
        df_list.append(loop2df(loop))

    return df_list


def star2df(starfile):
    with open(starfile) as f:
        star =[l for l in (line.strip() for line in f) if l and not l.startswith('#')] # read only non-blank and non "#" lines and rm all '\n' or spaces

    blockcode_idx = [i for i, x in enumerate(star) if x.startswith('data_')]
    blockcodes = [star[i] for i in blockcode_idx]
    blockcode_idx.append(len(star))
    blocks = [star[blockcode_idx[i]:blockcode_idx[i+1]] for i in range(len(blockcode_idx)-1)]

    block_list = []
    for block in blocks:
        block_list.append(block2df(block))

    star_df = dict(zip(blockcodes, block_list))
    return star_df


def df2loop(df, file):
    file.write('loop_ \n')

    keys = df.columns.tolist()
    for l in keys:
        file.write(l + ' \n')

    for i in range(len(df)):
        s = '  '.join(df.iloc[i].tolist())
        file.write(s + ' \n')

    file.write('\n')


def df2star(star_df, star_name):
    blockcodes = list(star_df.keys())
    block_list = list(star_df.values())

    with open(star_name, 'w') as f:
        for i in range(len(blockcodes)):
            f.write(blockcodes[i] + ' \n\n')
            df_list = block_list[i]
            for j in range(len(df_list)):
                df = df_list[j]
                df2loop(df, f)


def micBlockcode(star_df):
    if len(list(star_df.keys())) == 1:
        return list(star_df.keys())[0]
    else:
        return 'data_micrographs'

def star2miclist(starfile):
    star_df = star2df(starfile)
    mic_blockcode = micBlockcode(star_df)
    micList = star_df[mic_blockcode][0]['_rlnMicrographName'].tolist()
    return micList
