import java.util.*
import kotlin.collections.HashSet
import kotlin.math.max

// 802. 找到最终的安全状态
fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
    val stateList = IntArray(graph.size) { 0 }
    for (i in graph.indices) {
        eventualSafeNodesIsSafe(i, graph,stateList)
    }
    val ans = mutableListOf<Int>()
    for (i in stateList.indices) {
        if (stateList[i] == 2) ans.add(i)
    }
    return ans
}
fun eventualSafeNodesIsSafe(thisNode: Int, graph: Array<IntArray>, state: IntArray):Boolean {
    if (state[thisNode] != 0) return state[thisNode] == 2
    state[thisNode] = 1
    for (nextNode in graph[thisNode]) {
        if (!eventualSafeNodesIsSafe(nextNode, graph, state)) {
            return false
        }
    }
    state[thisNode] = 2
    return true
}
// 847. 访问所有节点的最短路径
fun shortestPathLength(graph: Array<IntArray>): Int {
    data class State(val thisNode: Int, val thisState: Int, val dis: Int)
    val n = graph.size
    val queue: Queue<State> = LinkedList()
    val hashSet = HashSet<Pair<Int, Int>>()
    for (i in graph.indices) {
        val firstState = State(i, 1 shl i, 0)
        queue.add(firstState)
        hashSet.add(Pair(firstState.thisNode, firstState.thisState))
    }
    while (queue.isNotEmpty()) {
        val headState = queue.remove()
        if (headState.thisState == ((1 shl n) -1)) { // 找到了
            return headState.dis
        }
        for (nextNode in graph[headState.thisNode]) {
            val nextState = State(nextNode, headState.thisState or (1 shl nextNode), headState.dis + 1)
            val hash = Pair(nextState.thisNode, nextState.thisState)
            if (!hashSet.contains(hash)) {
                queue.add(nextState)
                hashSet.add(hash)
            }
        }
    }
    return 0
}
// 413. 等差数列划分
fun numberOfArithmeticSlices(nums: IntArray): Int {
    if (nums.size < 3) return 0
    var idx = 2
    var gap = nums[1] - nums[0]
    var cnt = 2
    var ans = 0
    while (idx <= nums.size - 1) {
        if (gap == Int.MAX_VALUE) { // 重新开始
            if (idx <= nums.size - 2) { // 还可以重新开始
                gap = nums[idx] - nums[idx - 1]
                idx++
                cnt = 2
            }
            else {
                break
            }
        }
        else { // 已经有了
            val thisGap = nums[idx] - nums[idx - 1]
            if (thisGap == gap) {
                cnt++
                idx++
            }
            else { // 断了
                if (cnt >= 3) {
                    ans += (cnt - 1)*(cnt - 2) / 2
                }
                gap = Int.MAX_VALUE
            }
        }
    }
    if (cnt >= 3) {
        ans += (cnt - 1)*(cnt - 2) / 2
    }
    return ans
}
// 516. 最长回文子序列
fun longestPalindromeSubseq(s: String): Int {
    val n = s.length
    val dp = Array(n) { IntArray(n) }
    repeat(n) {
        dp[it][it] = 1
    }
    for (length in 2..n) {
        for (i in 0..n-length) {
            val j = i + length - 1
            if(s[i] == s[j]) {
                dp[i][j] = dp[i+1][j-1] + 2
            }
            else {
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
            }
        }
    }
    return dp[0][n-1]
}
// 526. 优美的排列
fun countArrangement(n: Int): Int {
    var ans = 0
    val array = IntArray(n + 1)
    val vis = IntArray(n + 1)

    return countArrangementDFS(vis, 1, n)
}
fun countArrangementDFS(vis: IntArray, deep: Int, n: Int): Int {
    var ansCountInThisDeep = 0
    if (deep == n + 1) {
        return 1
    }
    for (i in 1..n) {
        if (vis[i] == 0 && (i % deep == 0 || deep % i == 0)) {
            vis[i] = 1
            ansCountInThisDeep += countArrangementDFS(vis, deep + 1, n)
            vis[i] = 0
        }
    }
    return ansCountInThisDeep
}
// 551. 学生出勤记录 I
fun checkRecord(s: String): Boolean {
    var continuation = 0
    var absentDay = 0
    for (c in s) {
        if (c == 'A') {
            if (absentDay == 1) return false
            else {
                absentDay = 1
                continuation = 0
            }
        }
        else if (c == 'L') {
            if (continuation == 2) return false
            else continuation++
        }
        else {
            continuation = 0
        }
    }
    return true
}
fun main () {
    val array = intArrayOf(3)
    val intArray1 = intArrayOf(2,3,7,6)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1), intArrayOf(0,2,4), intArrayOf(1,3,4), intArrayOf(2), intArrayOf(1,2))
    val ans = countArrangement(14)
    println(ans)
}