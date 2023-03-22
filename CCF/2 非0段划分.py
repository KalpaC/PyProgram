# 作者 Ajex
# 创建时间 2023/3/18 16:55
# 文件名 2 非0段划分.py

# 意思是最长非零段
# 将数组A中【小于】p的数都变为0，求使非零段最多的p是多少

# 首先考虑贪心思路。可以先找出所有的非零段，然后对于长度不为1的非零段，看最多可以分成多少个非零段。
# 但是使一个非零段分裂的过程中可能导致其他非零段消失，究竟能否使非零段增多是不确定的。

# 有没有可能是二维动规？应该不是
# 直接暴力解
n = int(input())
A = tuple(map(int, input().split()))
low = 2100000000
for a in A:
    if a != 0 and a < low:
        low = a
up = max(A)+1


def non_zero_segment(arr, p):
    cnt = 0
    flag = 0
    for i in range(len(arr)):
        if (i == 0 or arr[i - 1] == 0 or arr[i - 1] < p) and arr[i] >= p:
            flag = 1
        elif arr[i] < p or arr[i] == 0:
            if flag == 1:
                cnt += 1
            flag = 0
    if flag == 1:
        cnt += 1
    return cnt


MAX = 0
for p in range(low, up):
    ans = non_zero_segment(A, p)
    if ans > MAX:
        MAX = ans
print(MAX)
