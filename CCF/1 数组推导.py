# 作者 Ajex
# 创建时间 2023/3/18 16:47
# 文件名 1 数组推导.py

# 如果B数组不变，则Ai可能为最大值，也可能为1，

n = int(input())
B = tuple(map(int, input().split()))
lastA = B[0]
maxSum = B[0]
minSum = B[0]
for i in range(1, len(B)):
    if B[i] == B[i - 1]:
        maxSum += lastA
        minSum += 0
    else:
        maxSum += B[i]
        minSum += B[i]
        lastA = B[i]
print(maxSum)
print(minSum)
