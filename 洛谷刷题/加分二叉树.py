# 10：11

# 给定了一个中序遍历为1~n的具体结构未知的二叉树
# 求其得分最高的一种二叉树，并输出分数，以及二叉树的前序遍历。

# 错因1：没认真读题，左子*右子+根，而不是左+右

# 于是显然没法贪心了，考虑搜索或dp。
# dp[1][n]表示1~n的中序遍历中的最大者
# dp[1][n] = max(dp[1][k-1]*dp[k+1][n]+score[k])

# 计算方向：与以往的不同，树形dp应该按照区间长度进行遍历求值

n = int(input())
score = [0] + list(map(int, input().split()))
dp = [None]
root = [None]
for _ in range(n):
    dp.append([0] * (n + 1))
    root.append([0] * (n + 1))


def recursivePrint(i, j):
    if i > j:
        return
    print(root[i][j],end=' ')
    recursivePrint(i, root[i][j] - 1)
    recursivePrint(root[i][j] + 1, j)


for span in range(0, n):
    for i in range(1, n - span + 1):
        # i+span<=n, i<=n-span
        if span == 0:
            dp[i][i] = score[i]
            root[i][i] = i
            continue
        for k in range(i, i + span + 1):
            mid = score[k]
            left = 1
            right = 1
            if k - 1 >= i:
                left = dp[i][k - 1]
            if k + 1 <= i + span:
                right = dp[k + 1][i + span]
            if left * right + mid > dp[i][i + span]:
                dp[i][i + span] = left * right + mid
                root[i][i + span] = k
print(dp[1][n])
recursivePrint(1,n)
