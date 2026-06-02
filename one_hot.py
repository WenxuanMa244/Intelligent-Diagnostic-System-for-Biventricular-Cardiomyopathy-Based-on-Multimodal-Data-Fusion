import numpy as np

# 定义一个独热编码向量
one_hot = np.array([1, 0, 0])

# 执行 1.0 - one_hot 操作
result = 1.0 - one_hot

print("独热编码向量:", one_hot)
print("1.0 - one_hot 的结果:", result)