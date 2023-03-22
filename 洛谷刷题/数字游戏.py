# 14:20

# 一圈共n个数字，将其分为m个部分，使每部分的和mod10再相乘的结果最大或最小。
# 这个问题和K个乘法很像，主要不同在于这个的收尾可以连接

# 处理圆圈的思路:开二倍的空间，限制总长度（限制长度会导致增加一维，没有必要）
# 更好的思路是开二倍空间，然后枚举起始下标

# 思路一：
# 0~n-1的最优值可以分解成(0~j的最优值*(j+1到n-1)的和mod10)对于任意属于0~n-1之间的j的最优值。
# 注意还需要插入一个分段次数k
# 于是有dp[i][k]表示以i为结束下标的分k段的最优值。
# 显然要求的值是dp[n-1][m]
# 计算方向：先算分段最少的，才能算出来分段多的。
# 段内应该i从左向右遍历

# 优化，先计算出前缀和，降低求和的时间复杂度
# 错误！！目前未知错因，但这个代码整体的错误概率确实很高，改了很多处代码还没改完
# 仍有一个点不过 最后发现是i的范围有误。一定要注意动规中变量的范围！！一定要结合物理含义细致分析。
n, m = tuple(map(int, input().split()))
num = []
sumMod10 = []
for i in range(n):
    num.append(int(input()))
    if i == 0:
        sumMod10.append(num[0] % 10)
    else:
        sumMod10.append((sumMod10[-1] + num[-1]) % 10)
for i in range(n - 1):
    num.append(num[i])  # 只需要写入n-1个即可
    sumMod10.append((sumMod10[-1] + num[-1]) % 10)
MAX = 0
MIN = 1000
dpMax = []
dpMin = []
for i in range(n):
    dpMax.append([0] * (n + 1))
    dpMin.append([1000] * (n + 1))


def getSum(i, j):
    if i < 1:
        return sumMod10[j]
    return (sumMod10[j] - sumMod10[i - 1]) % 10


for s in range(0, n):
    for i in range(0, n):
        dpMax[i][1] = getSum(s, s + i)
        dpMin[i][1] = getSum(s, s + i)
    for k in range(2, m + 1):
        # 要将0~i分为k段，k已定。共i+1个数，最多可分i+1段，所以i+1应该大于等于k，i>=k-1，
        for i in range(k-1, n):
            # 对于分割j来说，左侧0~j最多可分为j+1段，所以j+1>=k-1，j>=k-2。右侧，至少得有一个元素，也就是j<i
            g = 0
            l = 1000
            for j in range(k - 2, i):
                a = dpMax[j][k - 1] * getSum(s + j + 1, s + i)
                b = dpMin[j][k - 1] * getSum(s + j + 1, s + i)
                if a > g:
                    g = a
                if b < l:
                    l = b
            dpMax[i][k] = g
            dpMin[i][k] = l
    if dpMax[n - 1][m] > MAX:
        MAX = dpMax[n - 1][m]
    if dpMin[n - 1][m] < MIN:
        MIN = dpMin[n - 1][m]
print(MIN)
print(MAX)
