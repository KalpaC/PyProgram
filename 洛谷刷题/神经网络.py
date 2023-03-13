# 8:47

# 每个神经元具有初始值，或从上一层神经元获得值C。
# 每层神经元计算自己的C，（根据上一层的神经元输出和边的权值加权再减去阈值得到本层C）
# 最终算出最终结果

# 本题显然可以暴力模拟，但时间复杂度比较高。

# 思路一：对神经网络进行拓扑排序，按照拓扑排序的顺序依次计算神经元状态。
# 需要的数据结构：队列或栈。内部元素应该为编号、入度的二元组。在进行拓扑排序的时候，对下一层的元素顺便修改。
# 图采用何种方式保存？二维数组：不合适，空间足够，但是修改入度时需要O(V)
# 邻接表：合适，但需要自己写节点，当然也不算难。本题采用这种方式。
# 错点：没认真读题，状态小于等于0的神经元不会参与下一次的计算
class Node:
    def __init__(self):
        self.inDegree = 0
        self.edges = []  # 格式：(节点编号，权值)
        self.u = 0
        self.state = 0


nodes = [None]
stack = []
ans = []
n, p = tuple(map(int, input().split()))
for i in range(n):
    s, u = tuple(map(int, input().split()))
    Ni = Node()
    Ni.state = s
    Ni.u = u
    nodes.append(Ni)
for _ in range(p):
    i, j, w = tuple(map(int, input().split()))
    nodes[i].edges.append((j, w))
    nodes[j].inDegree += 1
# 统计入度为0的节点，入度为0意味着计算完毕，可以减去阈值。初始的不需要
for i in range(1, n + 1):
    if nodes[i].inDegree == 0:
        stack.append(i)
while len(stack) > 0:
    i = stack.pop()
    for j, w in nodes[i].edges:
        nodes[j].inDegree -= 1
        if nodes[i].state > 0:
            nodes[j].state += nodes[i].state * w
        if nodes[j].inDegree == 0:
            nodes[j].state -= nodes[j].u
            stack.append(j)
    if len(nodes[i].edges) == 0:
        ans.append((i, nodes[i].state))
ans.sort()
cnt = 0
for i, s in ans:
    if s > 0:
        print(i, s)
        cnt += 1
if cnt == 0:
    print('NULL')
# 思路二：根据输入确定输入层的初始态。然后进行广度优先搜索，对于已知状态的神经元，搜索其下一层的神经元，对其Ci进行修改。
# 为了确定神经元的状态已知，需要按层次遍历。
