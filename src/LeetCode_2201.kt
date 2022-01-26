import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashSet
import kotlin.collections.ArrayDeque
import kotlin.math.abs

// 71. 简化路径
fun simplifyPath(path: String): String {
    val dirs = path.split('/')
    val stack: Deque<String> = LinkedList()
    for (dir in dirs) {
        if (dir == "..") {
            if (stack.isNotEmpty()) stack.removeLast()
        } else if (dir == "." || dir == "") {
            continue
        } else {
            stack.addLast(dir)
        }
    }
    if (stack.isEmpty()) return "/"
    var ans = ""
    while (stack.isNotEmpty()) {
        ans += "/${stack.first()}"
        stack.removeFirst()
    }
    return ans
}

// 1614. 括号的最大嵌套深度
fun maxDepth(s: String): Int {
    var ans = 0
    var depth = 0
    for (c in s) {
        when (c) {
            '(' -> {
                depth++
                ans = if (depth > ans) depth else ans
            }
            ')' -> {
                depth--
            }
        }
    }
    return ans
}

// 89. 格雷编码
fun grayCodeDfs(ans: MutableList<Int>, n: Int, vis: HashSet<Int>): Boolean {
    if (1.shl(n) == ans.size) return true
    for (i in 0 until n) {
        val next = ans.last().xor(1.shl(i))
        if (!vis.contains(next)) {
            ans.add(next)
            vis.add(next)
            if (grayCodeDfs(ans, n, vis)) return true
        }
    }
    return false
}

fun grayCode(n: Int): List<Int> {
    val ans = ArrayList<Int>(1.shl(n))
    ans.add(0)
    val vis = hashSetOf(0)
    grayCodeDfs(ans, n, vis)
    return ans
}

// 1629. 按键持续时间最长的键
fun slowestKey(releaseTimes: IntArray, keysPressed: String): Char {
    var ans = Pair(keysPressed[0], releaseTimes[0])
    for (i in 1 until releaseTimes.size) {
        val pressTime = releaseTimes[i] - releaseTimes[i - 1]
        val key = keysPressed[i]
        if (ans.second < pressTime || ans.second == pressTime && ans.first < key) ans = Pair(key, pressTime)
    }
    return ans.first
}

// 306. 累加数
fun isAdditiveNumber(num: String): Boolean {
    val upper = if (num.length % 2 == 0) num.length / 2 - 1 else num.length / 2
    for (i in 0 until upper) {
        if (num[0] == '0' && i != 0) break
        val firstNum = num.slice(0..i)
        for (j in i + 1 until num.length - 1) {
            if (num[i + 1] == '0' && j != i + 1) break
            var secondNum = num.slice(i + 1..j)
            var nxt = (firstNum.toLong() + secondNum.toLong()).toString()
            var begin = j + 1
            while (begin + nxt.length <= num.length) {
                if (num.slice(begin until begin + nxt.length).toLong() == nxt.toLong()) {
                    begin += nxt.length
                    val tmp = secondNum.toLong()
                    secondNum = nxt
                    nxt = (tmp + nxt.toLong()).toString()
                } else {
                    break
                }
            }
            if (begin == num.length) return true
        }
    }
    return false
}

// 1036. 逃离大迷宫
fun isEscapePossibleJudgeNext(
    now: Pair<Int, Int>,
    direction: Int,
    blocked: Set<Pair<Int, Int>>,
    vis: MutableSet<Pair<Int, Int>>
): Pair<Int, Int>? {
    val nexts = arrayOf(intArrayOf(-1, 0), intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1))
    val next = Pair(now.first + nexts[direction][0], now.second + nexts[direction][1])
    if (next.first < 0 || next.second < 0 || next.first >= 1e6 || next.second >= 1e6) return null
    if (blocked.contains(next) || vis.contains(next)) return null
    vis.add(next)
    return next
}

fun isEscapePossibleCheck(
    handledBlocked: Set<Pair<Int, Int>>,
    magicMax: Int,
    source: IntArray,
    target: IntArray
): Boolean {
    val queue: MutableList<Pair<Int, Int>> = ArrayDeque()
    val vis = mutableSetOf<Pair<Int, Int>>()
    queue.add(Pair(source[0], source[1]))
    vis.add(Pair(source[0], source[1]))
    while (queue.isNotEmpty()) {
        val now = queue.removeFirst()
        repeat(4) {
            isEscapePossibleJudgeNext(now, it, handledBlocked, vis)?.let { next ->
                if (next == Pair(target[0], target[1]) || vis.size > magicMax) return true
                else queue.add(next)
            }
        }
    }
    return false
}

fun isEscapePossible(blocked: Array<IntArray>, source: IntArray, target: IntArray): Boolean {
    val handledBlocked = blocked.map {
        Pair(it[0], it[1])
    }.toSet()
    val magicMax = blocked.size * (blocked.size - 1) / 2
    return isEscapePossibleCheck(handledBlocked, magicMax, source, target)
            && isEscapePossibleCheck(handledBlocked, magicMax, target, source)
}

// 334. 递增的三元子序列
fun increasingTripletSolution1(nums: IntArray): Boolean {
    val leftMin = IntArray(nums.size) { it }
    val rightMax = IntArray(nums.size) { it }
    var minIdx = 0
    for (i in nums.indices) {
        if (nums[i] > nums[minIdx]) {
            leftMin[i] = minIdx
        } else {
            minIdx = i
        }
    }
    var maxIdx = nums.lastIndex
    for (i in nums.lastIndex downTo 0) {
        if (nums[i] < nums[maxIdx]) {
            rightMax[i] = maxIdx
        } else {
            maxIdx = i
        }
    }
    for (i in nums.indices) {
        if (leftMin[i] != i && rightMax[i] != i) return true
    }
    return false
}

fun increasingTriplet(nums: IntArray): Boolean {
    if (nums.size < 3) return false
    val bigger = IntArray(nums.size) { -1 }
    val stack = mutableListOf<Pair<Int, Int>>()
    for (i in nums.indices) {
        val next = nums[i]
        while (stack.isNotEmpty() && stack.last().first < next) {
            val top = stack.removeAt(stack.size - 1)
            bigger[top.second] = i
        }
        stack.add(Pair(next, i))
    }
    val vis = IntArray(nums.size) { 0 }
    for (i in bigger.indices) {
        if (vis[i] != 1) {
            var iter = i
            var cnt = 1
            while (bigger[iter] != -1) {
                iter = bigger[iter]
                vis[iter] = 1
                cnt++
                if (cnt >= 3) return true
            }
        }
    }
    return false
}

// 747. 至少是其他数字两倍的最大数
fun dominantIndex(nums: IntArray): Int {
    var maxIdx = -1
    var max = -1
    var subMax = -1
    for (i in nums.indices) {
        val now = nums[i]
        if (now > max) {
            subMax = max
            max = now
            maxIdx = i
        } else if (now > subMax) subMax = now
    }
    return if (max >= subMax * 2)
        maxIdx
    else
        -1
}

// 373. 查找和最小的K对数字
fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> {
    val comparator = compareBy<Pair<Int, Int>> { nums1[it.first] + nums2[it.second] }
    val priorityQueue = PriorityQueue(k, comparator)
    val ans = mutableListOf<List<Int>>()
    for (i in nums1.indices) {
        priorityQueue.add(Pair(i, 0))
    }
    while (ans.size < k && priorityQueue.isNotEmpty()) {
        val now = priorityQueue.poll()
        ans.add(listOf(nums1[now.first], nums2[now.second]))
        if (now.second + 1 < nums2.size) priorityQueue.add(Pair(now.first, now.second + 1))
    }
    return ans
}

// 1716. 计算力扣银行的钱
fun totalMoney(n: Int): Int {
    var mondayMoney = 1
    val fullWeek = n / 7
    var ans = 0
    repeat(fullWeek) {
        ans += (mondayMoney * 2 + 6) * 7 / 2
        mondayMoney++
    }
    val remainDay = n % 7
    ans += (mondayMoney * 2 + remainDay - 1) * remainDay / 2
    return ans
}

// 382. 链表随机节点
class Solution(private val head: ListNode?) {
    fun getRandom(): Int {
        var ans = head?.`val`
        var idx = 1
        var current = head?.next
        while (current != null) {
            val random = (0..idx).random()
            if (random < 1) ans = current.`val`
            idx++
            current = current.next
        }
        return ans ?: 0
    }
}

// 1220. 统计元音字母序列的数目
fun countVowelPermutationSolution1(n: Int): Int {
    val MOD = (1e9 + 7).toLong()
    val dp = Array(2) { idx -> MutableList<Long>(5) { if (idx == 0) 1 else 0 } }
    repeat(n - 1) {
        dp[1][0] = (dp[0][1] + dp[0][2] + dp[0][4]) % MOD // a0: e1, i2, u4
        dp[1][1] = (dp[0][0] + dp[0][2]) % MOD // e1: a0, i2,
        dp[1][2] = (dp[0][1] + dp[0][3]) % MOD  // i2: e1, o3
        dp[1][3] = dp[0][2] // o3: i2
        dp[1][4] = (dp[0][2] + dp[0][3]) % MOD // u4: i2, o3
        dp[0] = dp[1].toMutableList()
    }
    return dp[0].reduce { sum, it -> (sum + it) % MOD }.toInt()
}

fun matrixMul(a: List<List<Long>>, b: List<List<Long>>, MOD: Long): List<List<Long>> {
    val m = a.indices
    val n = a[0].indices
    val k = b[0].indices
    val ans = MutableList<MutableList<Long>>(0) { mutableListOf() }
    for (rowA in m) {
        val rowC = mutableListOf<Long>()
        for (colB in k) {
            var sum: Long = 0
            for (i in n) {
                sum = (sum + (a[rowA][i] * b[i][colB]) % MOD) % MOD
            }
            rowC.add(sum)
        }
        ans.add(rowC)
    }
    return ans
}

fun countVowelPermutationSolution2(n: Int): Int {
    val MOD = (1e9 + 7).toLong()
    var m: List<List<Long>> = listOf(
        listOf(0, 1, 1, 0, 1),
        listOf(1, 0, 1, 0, 0),
        listOf(0, 1, 0, 1, 0),
        listOf(0, 0, 1, 0, 0),
        listOf(0, 0, 1, 1, 0)
    )
    var time = n - 1
    var ans: List<List<Long>> = listOf(listOf(1), listOf(1), listOf(1), listOf(1), listOf(1))
    while (time != 0) {
        if (time.and(1) == 1) {
            ans = matrixMul(m, ans, MOD)
        }
        m = matrixMul(m, m, MOD)
        time = time.shr(1)
    }
    return ans.flatten().reduce { sum, it -> (sum + it) % MOD }.toInt()
}

// 539. 最小时间差
fun findMinDifference(timePoints: List<String>): Int {
    if (timePoints.size > 24 * 60) return 0
    val sortedList = timePoints
        .map {
            val splitTime = it.split(':')
            splitTime[0].toInt() * 60 + splitTime[1].toInt()
        }
        .sorted()
    return sortedList.foldIndexed(24 * 60 - sortedList.last() + sortedList.first()) { index, min, _ ->
        if (index == 0) min
        else {
            min.coerceAtMost(sortedList[index] - sortedList[index - 1])
        }
    }
}

// 300. 最长递增子序列
fun lengthOfLISSolution1(nums: IntArray): Int {
    val ans = mutableListOf<Int>()
    nums.forEach {
        if (ans.isEmpty() || ans.last() < it) {
            ans.add(it)
        } else {
            val index = ans.binarySearch(it)
            if (index < 0) ans[-index - 1] = it
        }
    }
    return ans.size
}

fun lengthOfLISSolution2(nums: IntArray): Int {
    val dp = MutableList(nums.size) { 1 }
    var ans = 1
    nums.forEachIndexed { numsIdx, num ->
        var maxDp = 0
        for (i in 0 until numsIdx) {
            if (nums[i] < nums[numsIdx]) maxDp = Math.max(maxDp, dp[i])
        }
        dp[numsIdx] = maxDp + 1
        ans = Math.max(ans, dp[numsIdx])
    }
    return ans
}

// 219. 存在重复元素 II
data class Ele(val idx: Int, val num: Int) : Comparable<Ele> {
    override fun compareTo(other: Ele) = when {
        num != other.num -> num - other.num
        else -> idx - other.idx
    }
}

fun containsNearbyDuplicate(nums: IntArray, k: Int): Boolean {
    val priorityQueue = PriorityQueue<Ele>()
    nums.forEachIndexed { idx, num ->
        priorityQueue.add(Ele(idx, num))
    }
    var lastEle = priorityQueue.poll()
    while (priorityQueue.isNotEmpty()) {
        val now = priorityQueue.poll()
        if (now.num == lastEle.num && now.idx - lastEle.idx <= k) return true
        lastEle = now
    }
    return false
}

// 2029. 石子游戏 IX
fun stoneGameIX(stones: IntArray): Boolean {
    val nums = mutableListOf(0, 0, 0)
    for (stone in stones) {
        nums[stone % 3]++
    }
    return if (nums[0] % 2 == 0) {
        nums[1] != 0 && nums[2] != 0
    } else {
        abs(nums[1] - nums[2]) > 2
    }
}

// 1345. 跳跃游戏 IV
data class MinJumpsPair(val idx: Int, val depth: Int)

fun minJumps(arr: IntArray): Int {
    val map = mutableMapOf<Int, ArrayDeque<Int>>()
    val vis = mutableSetOf<Int>() // idx
    val queue = ArrayDeque<MinJumpsPair>()
    arr.forEachIndexed { idx, ele ->
        if (idx == 0 || idx == arr.size - 1 || arr[idx - 1] != arr[idx] || arr[idx + 1] != arr[idx]) {
            if (map[ele] != null) map[ele]?.addFirst(idx)
            else map[ele] = ArrayDeque(listOf(idx))
        }
    }
    queue.addLast(MinJumpsPair(0, 0))
    vis.add(0)
    while (queue.isNotEmpty()) {
        val now = queue.removeFirst()
        if (now.idx == arr.size - 1) return now.depth
        map[arr[now.idx]]?.let {
            it.forEach { nxtIdx ->
                if (nxtIdx == arr.size - 1) return now.depth + 1
                if (!vis.contains(nxtIdx)) {
                    queue.addLast(MinJumpsPair(nxtIdx, now.depth + 1))
                    vis.add(nxtIdx)
                }
            }
            map.remove(arr[now.idx])
        }
        if (now.idx < arr.size - 1 && !vis.contains(now.idx + 1)) {
            if (now.idx + 1 == arr.size - 1) return now.depth + 1
            queue.addLast(MinJumpsPair(now.idx + 1, now.depth + 1))
            vis.add(now.idx + 1)
        }
        if (now.idx > 0 && !vis.contains(now.idx - 1)) {
            queue.addLast(MinJumpsPair(now.idx - 1, now.depth + 1))
            vis.add(now.idx - 1)
        }
    }
    return arr.size - 1
}

// 1332. 删除回文子序列
fun removePalindromeSub(s: String): Int {
    if (s.isEmpty()) return 0
    for (i in 0 until s.length / 2) {
        if (s[i] != s[s.length - 1 - i]) return 2
    }
    return 1
}

// 2034. 股票价格波动
class StockPrice() {
    private val prices = hashMapOf<Int, Int>() // timestamp, price
    private val orders = TreeMap<Int, Int>() // price, cnt
    private var current = 0
    fun update(timestamp: Int, price: Int) {
        if (prices.containsKey(timestamp)) { // 修正
            val oldPrice: Int = prices[timestamp]!!
            if (orders[oldPrice] == 1) {
                orders.remove(oldPrice)
            } else {
                orders[oldPrice] = orders[oldPrice]!!.minus(1)
            }
        }
        current = Math.max(timestamp, current)
        prices[timestamp] = price
        orders[price] = orders[price]?.plus(1) ?: 1
    }

    fun current(): Int {
        return prices[current]!!
    }

    fun maximum(): Int {
        return orders.lastKey()
    }

    fun minimum(): Int {
        return orders.firstKey()
    }
}

// 2045. 到达目的地的第二短时间
data class SecondMinimumPair(val idx: Int, val depth: Int)

fun secondMinimum_CalTime(num: Int, passTime: Int, change: Int): Int {
    var ret = 0
    repeat(num) {
        if (ret % (2 * change) >= change) {
            ret += (2 * change - ret % (2 * change))
        }
        ret += passTime
    }
    return ret
}

fun secondMinimum(n: Int, edges: Array<IntArray>, time: Int, change: Int): Int {
    val nxtNodeTable = mutableMapOf<Int, MutableList<Int>>()
    val lastNode = MutableList(n + 1) { it }
    edges.forEach {
        if (!nxtNodeTable.containsKey(it[0])) {
            nxtNodeTable[it[0]] = mutableListOf(it[1])
        } else {
            nxtNodeTable[it[0]]!!.add(it[1])
        }
        if (!nxtNodeTable.containsKey(it[1])) {
            nxtNodeTable[it[1]] = mutableListOf(it[0])
        } else {
            nxtNodeTable[it[1]]!!.add(it[0])
        }
    }
    val visDepth = MutableList<MutableSet<Int>>(n + 1) { mutableSetOf() }
    val queue = ArrayDeque<SecondMinimumPair>()

    queue.add(SecondMinimumPair(1, 0))
    visDepth[1].add(0)

    while (queue.isNotEmpty()) {
        val now = queue.removeFirst()
        nxtNodeTable[now.idx]?.let {
            for (nxtIdx in it) {
                if (nxtIdx == n && visDepth[n].size == 1 && !visDepth[n].contains(now.depth + 1)) {
                    return secondMinimum_CalTime(now.depth + 1, time, change)
                }
                if (visDepth[nxtIdx].size != 2) {
                    queue.addLast(SecondMinimumPair(nxtIdx, now.depth + 1))
                    lastNode[nxtIdx] = now.idx
                    visDepth[nxtIdx].add(now.depth + 1)
                }
            }
        }
    }
    return -1
}

// 1688. 比赛中的配对次数
fun numberOfMatches(n: Int): Int {
    var remainTeam = n
    var ans = 0
    while (remainTeam > 1) {
        ans += remainTeam / 2
        remainTeam = remainTeam / 2 + remainTeam % 2
    }
    return ans
}

// 2013. 检测正方形
class DetectSquares() {
    private val map = mutableMapOf<Int, MutableMap<Int, Int>>()
    fun add(point: IntArray) {
        val x = point[0]
        val y = point[1]
        map.putIfAbsent(x, mutableMapOf())
        map[x]!![y] = map[x]!!.getOrDefault(y, 0) + 1
    }

    fun count(point: IntArray): Int {
        var ans = 0
        val x = point[0]
        val y = point[1]
        val xCol = map.getOrDefault(x, mutableMapOf())
        for (p1 in xCol) {
            val length = y - p1.key
            if (length == 0) continue
            ans += p1.value * (map[x - length]?.get(p1.key) ?: 0) * (map[x - length]?.get(y) ?: 0)
            ans += p1.value * (map[x + length]?.get(p1.key) ?: 0) * (map[x + length]?.get(y) ?: 0)
        }
        return ans
    }
}

fun main() {
    val num = 3
    val array = arrayOf(
        intArrayOf(1, 2),

        )
    val ans = secondMinimum(2, array, 3, 2)
    println(ans)
}