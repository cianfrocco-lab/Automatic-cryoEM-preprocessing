'''
Helper functions to for data prep.
'''

def star2df(starfile):
    with open(starfile) as f:
        star = f.readlines()

    for i in range(len(star)):
        if 'loop_' in star[i]:
            start_idx = i
            break
    key_idx = []
    for j in range(start_idx+1, len(star)):
        if star[j].startswith('_'):
            key_idx.append(j)

    keys = [star[ii] for ii in key_idx]
    star_df = star[1+key_idx[-1]:]
    star_df = [x.split() for x in star_df]
    star_df = pd.DataFrame(star_df)
    star_df = star_df.dropna()
    star_df.columns = keys
    micname_key = [x for x in keys if 'MicrographName' in x][0]

    return star_df, micname_key


def df2star(star_df, star_name):
    header = ['data_ \n', '\n', 'loop_ \n']
    keys = star_df.columns.tolist()

    with open(star_name, 'w') as f:
        for l_0 in header:
            f.write(l_0)
        for l_1 in keys:
            f.write(l_1)
        for i in range(len(star_df)):
            s = '  '.join(star_df.iloc[i].tolist())
            f.write(s + ' \n')


def star2miclist(starfile):
    with open(starfile) as f:
        star = f.readlines()

    for i in range(len(star)):
        if 'loop_' in star[i]:
            start_idx = i
            break
    key_idx = []
    for j in range(start_idx+1, len(star)):
        if star[j].startswith('_'):
            key_idx.append(j)

    keys = [star[ii] for ii in key_idx]
    star_df = star[1+key_idx[-1]:]
    star_df = [x.split() for x in star_df]
    star_df = pd.DataFrame(star_df)
    star_df = star_df.dropna()
    star_df.columns = keys
    micname_key = [x for x in keys if 'MicrographName' in x][0]
    mic_list = star_df[micname_key].tolist()

    return mic_list
