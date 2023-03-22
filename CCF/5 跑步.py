# 作者 Ajex
# 创建时间 2023/3/19 10:30
# 文件名 5 跑步.py

# 有三种事件：
#   1. 添加一个队员，强度为x；或者速度末位淘汰
#   2. s天结束时区间l,r的强度变为y倍
#   3. 统计第s天结束时l,r选手强度之和mod p

# 数据一定会溢出，所以根据mod p操作，做如下处理：
# (x1+x2)%p = (x1%p+x2%p)%p，所以每个数保存时都可以%p
# (x*y)%p = (x%p*y*p)%p，所以不单每个数保存时可以%p，乘y倍时也可以%p
# 一眼线段树。如何写？维护sum%p的值。
# ((x1+x2+x3)*y)%p = ((x1+x2+x3)%p * y%p)%p = (((x1%p+x2%p)%p*y%p)%p + ((x3%p)*y%p)%p)%p

# 写个线段树吧

class Node:
    def __init__(self):
        self.left = 0
        self.right = 0
        self.sum_mod_p = 0
        self.lazy = False
tree = []
n = 0


