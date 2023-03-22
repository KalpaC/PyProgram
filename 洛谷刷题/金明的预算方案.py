# 作者 Ajex
# 创建时间 2023/3/16 16:39
# 文件名 金明的预算方案.py

# 这个题一眼背包，只不过如何将问题转化为标准的背包是一个难点

# 物品分为主件和附件。每个物品具有价格和重要度两个指标。最优值是价值*重要度最大，容量是总价值
# 如果购买一个附件，需要一并购买主件，这就意味着需要判断主件是否已购。
# 由于一个物品只需要买一个，所以是01背包的变种。

# 那么如何判断主件是否已购呢？
# 加入记录结构？（误，因为背包不是深搜，记忆能力全部体现在状态转移方程中）
# 修改状态转移方程
# dp[i][j]=max(dp[i-1][j],dp[i-1][j-v[i]]+v[i]*p[i],dp[i-1][j-v[1]-v[i的第一个附件]],,)
# 所以实际上是只考虑主件的，附件只作为主件是否加入的附加条件判断。

n, m = tuple(map(int, input().split()))
v = [0] * (m + 1)
p = [0] * (m + 1)
subs = {}
isMaster = [False] * (m + 1)
for i in range(1, m + 1):
    v[i], p[i], q = tuple(map(int, input().split()))
    if q == 0:
        isMaster[i] = True
    else:
        if q not in subs:
            subs[q] = []
        subs[q].append(i)
dp = [0] * (n + 1)
for i in range(1, m + 1):
    if not isMaster[i]:
        continue
    for j in range(n, v[i] - 1, -1):
        cases = [dp[j], dp[j - v[i]] + v[i] * p[i]]
        if i in subs:
            # 1
            va = v[subs[i][0]]
            pa = p[subs[i][0]]
            if j - v[i] - va >= 0:
                cases.append(dp[j - v[i] - va] + v[i] * p[i] + va * pa)
            # 2
            if len(subs[i]) == 2:
                vb = v[subs[i][1]]
                pb = p[subs[i][1]]
                if j - v[i] - vb >= 0:
                    cases.append(dp[j - v[i] - vb] + v[i] * p[i] + vb * pb)
                if j - v[i] - va - vb >= 0:
                    cases.append(dp[j - v[i] - va - vb] + v[i] * p[i] + va * pa + vb * pb)
        dp[j] = max(cases)
print(dp[n])
