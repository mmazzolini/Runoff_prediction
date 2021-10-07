import numpy as np


def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
    return train_index