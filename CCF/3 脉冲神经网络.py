# 作者 Ajex
# 创建时间 2023/3/18 17:23
# 完成时间 19:34
# 文件名 3 脉冲神经网络.py
import queue

# 先梳理题目信息
# 1. 有N个神经元，S个突触，P个脉冲源
# 2. 脉冲通过突触连接到神经元上，神经元之间也通过突触连接。所以可以把脉冲看作会主动发出脉冲的的突触。
# 3. 时间规则：时间是离散的，固定间隔deltaT。
# 4. 脉冲源释放规则：给了一个伪随机函数，以及一个阈值r。每个时刻脉冲源会更新随机数，如果r>随机数，则在当前时刻发出一个脉冲
# 5. 神经元：神经元具有状态：v和u，具有常量a，b，c，d。
#   在每个新的时间间隔，神经元统计自己在这个时间间隔接收到的脉冲强度之和，并按照公式计算新的v和u。
#   如果新的v大于30，则向自己连接的神经元突触发出脉冲。而后v恢复到c，u增加到u+d
# 6. 突触：突触分为两种，脉冲源-神经元突触，神经元-神经元突触。
#   突触具有方向、延迟。脉冲只会沿一个方向传递，并在延迟D之后到达出结点。

# 现在统计在每个时刻k开始时要做什么
# 1. 遍历脉冲源，更新随机数，判断是否发射脉冲源。如果发射脉冲源，修改突触k+D时刻的值。（没必要，突触只用来定位，直接修改神经元）
#   注意所有脉冲源调用的是同一个rand函数，所以所以一轮会让它更新好多次
# 2. 遍历神经元。首先检测接入神经元的突触（如何做到？）以此更新自己的u与v，然后判断v是否大于30，是否需要修改突触（神经元）
# 这种方法需要脉冲/入神经元-突触有哈希映射，出神经元也需要保存哪些突触连接到自己。

# 第二种方法只改变第二步，即先遍历突触，因为突触中肯定保存了出神经元，不需要给神经元中作单独的映射。
# 再遍历神经元。似乎第二种更好？

# 脉冲源不需要类
next = 1
N, S, P, T = tuple(map(int, input().split()))
deltaT = float(input())


def myrand():
    global next
    next = (next * 1103515245 + 12345) % (2 ** 64)
    return (next // 65536) % 32768


def appendUnits(rn, v, u, a, b, c, d):
    for _ in range(rn):
        units.append(Unit(v, u, a, b, c, d))


class Unit:
    def __init__(self, v, u, a, b, c, d):
        self.v, self.u, self.a, self.b, self.c, self.d = v, u, a, b, c, d
        self.I = 0
        self.cnt = 0


units = []  # 神经元是一个数组，

# 脉冲信息采用优先队列保存，而不在神经元中保存，因为没有必要保留那么久，浪费空间
pulses = queue.PriorityQueue()

connectMap = {}  # inID: [(outID, w, D),] int:list


def pulseSourceCheck(sid, t):
    rand = myrand()
    if units[sid] > rand:
        reliefPulse(sid, t)


def reliefPulse(inID: int, t):
    if inID in connectMap:
        for outID, w, D in connectMap[inID]:
            pulses.put((t + D, outID, w))


def pulseSpread(t):
    while not pulses.empty():
        tmp = pulses.get()
        if tmp[0] > t:
            pulses.put(tmp)
            break
        t_, outID, w = tmp
        units[outID].I += w


def updateUnit(uid, t):
    v, u = units[uid].v, units[uid].u
    # print(units[uid].I,deltaT,v,u)
    units[uid].v = v + deltaT * (0.04 * (v ** 2) + 5 * v + 140 - u) + units[uid].I
    units[uid].u = u + deltaT * units[uid].a * (units[uid].b * v - u)
    units[uid].I = 0
    # print('时刻%d，v的值为' % t, units[uid].v)
    if units[uid].v >= 30:
        reliefPulse(uid, t)
        units[uid].v = units[uid].c
        units[uid].u = units[uid].u + units[uid].d
        units[uid].cnt += 1


while len(units) < N:
    x = input().split()
    appendUnits(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]))

for _ in range(P):
    units.append(int(input()))
# print(units)
for _ in range(S):
    x = input().split()
    inID, outID, w, D = int(x[0]), int(x[1]), float(x[2]), int(x[3])
    if inID not in connectMap:
        connectMap[inID] = []
    connectMap[inID].append((outID, w, D))

for k in range(1, T + 1):
    pulseSpread(k)
    for sid in range(N, N + P):
        pulseSourceCheck(sid, k)
    for uid in range(0, N):
        updateUnit(uid, k)

minV = 2100000000
maxV = -2100000000
maxCnt = 0
minCnt = 2100000000
for uid in range(0, N):
    if units[uid].v > maxV:
        maxV = units[uid].v
    if units[uid].v < minV:
        minV = units[uid].v
    if units[uid].cnt > maxCnt:
        maxCnt = units[uid].cnt
    if units[uid].cnt < minCnt:
        minCnt = units[uid].cnt
print('%.3f %.3f' % (minV, maxV))
print(minCnt, maxCnt)
