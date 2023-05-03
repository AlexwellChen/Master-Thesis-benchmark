for index, row in df.iterrows():
#     # covert ndarray to float
#     # print((df.loc[index, 'time'] * cost_factor[df.loc[index, 'device']]))
#     if df.loc[index, 'device'] == 'A10':
#         df.loc[index, 'cost'] = df.loc[index, 'time'] * cost_factor[df.loc[index, 'device']] / 3600 * 0.8
#     # df.loc[index, 'cost'] = df.loc[index, 'time'] * cost_factor[df.loc[index, 'device']] / 3600
