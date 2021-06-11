import numpy as np

def split_x(seq, size): # sequance를 주소를 1씩 증가시키며 size만큼의 데이터로 나눈다. 
    aaa = []        
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) 
    print(type(aaa))
    return np.array(aaa)

raw_data = np.array(range(1, 11))
print(f'base : {raw_data}\n')

size = 5
dataset = split_x(raw_data, size) # (6, 5)
print(f'split_x : \n{dataset}\n')

x = dataset[:, :4] # (6, 4)
y = dataset[:, -1] # (6,)
print(f'x_data : \n{x}')
print(f'y_data : \n{y}')