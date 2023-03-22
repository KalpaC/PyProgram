# 作者 Ajex
# 创建时间 2023/3/17 15:58
# 文件名 细胞分裂.py

# 题意分析：有n个数，S1~Sn
# 对于Si，称使得(Si**k mod m1**m2)==0 最小的k为最少时间
# 找到对应的k最小的i

# 显然是个数学问题。其实就是要看一个Si能除多少个m1
# Si^k/m1^m2 = C
# Si^(m2*k/m2)/m1^m2 = (Si^(k/m2)/m1)^m2 = C
# 所以Si^(k/m2)/m1一定是个整数
# 将Si和m1分解为质因数级数序列，用m1序列除以Si序列，如果0/0没关系，X/0直接放弃，找到最大的X/Y，k/m2=max(X/Y)

# 能否加速？
# 直接用m1的质因数去试除Si，能整除则记录可以除几次k（找到最大的k），不能则i+=1，
# 这个是错的，因为用Si除m1的质因数只能找到

def getFactors(num):
    i = 2
    ans = []
    mid = num >> 1
    while i <= mid:
        while num % i == 0:
            num = num // i
            if len(ans) == 0 or i != ans[-1]:
                ans.append(i)
        i += 1
    return ans



n = int(input())
m1, m2 = tuple(map(int, input().split()))
Sn = tuple(map(int, input().split()))
# 首先给m1质因数分解
factor = getFactors(m1)
MIN = 2100000000
minI = -1
for i in range(n):
    MAX = 0
    for f in factor:
        if Sn[i] % f != 0:
            break
        x = f // Sn[i]
        if x > MAX:
            MAX = x

print(minI + 1)
