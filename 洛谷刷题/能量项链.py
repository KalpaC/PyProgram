# 17：03 限时半小时完成

# 典型的矩阵乘法问题，但是2维dp

# dp[i][j]表示下标从i~j的珠子合并的最大能量为dp[i][j]
# dp[i][j] = max(dp[i][k]+dp[k+1][j]+head[i]*head[k+1]*head[j+1])
# 显然按长度遍历
# 注意当i=j时，只有一个珠子，其本身没有能量dp[i][j]=0

# 重点一，如何处理输入的数据

N = int(input())
# 为了避免复杂的逻辑转换，我们让珠子的下标从1开始，到0结束
head = [0] + list(map(int, input().split()))
# 目前最后一位下标为N
# 取N个数即可
for i in range(1,N+1):
    head.append(head[i])

# 在不考虑环的情况下：
dp = []
MAX = 0

for _ in range(N + 1):
    dp.append([0] * (N + 1))

for offset in range(N):
    # print(head)
    for length in range(N):
        # i+length<=N
        for i in range(1, N - length + 1):
            j = i + length
            peak = 0
            for k in range(i, j):
                v = dp[i][k] + dp[k + 1][j] + head[i+offset] * head[k + 1+offset] * head[j + 1+offset]
                if v > peak:
                    peak = v
            dp[i][j] = peak
    if dp[1][N]>MAX:
        MAX = dp[1][N]
print(MAX)


