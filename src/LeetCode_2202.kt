import java.util.*
import kotlin.collections.ArrayDeque
import kotlin.math.max
import kotlin.math.min

// 395. 至少有 K 个重复字符的最长子串
fun longestSubstring(s: String, k: Int): Int {
    if (s.length < k) return 0
    val charCnt = mutableMapOf<Char, Int>()
    s.forEach { c ->
        charCnt[c] = charCnt.getOrDefault(c, 0) + 1
    }
    val chars = mutableSetOf<Char>()
    charCnt.forEach { (key, value) ->
        if (value < k) chars.add(key)
    }
    if (chars.size == 0) return s.length
    val subStrings = s.split(*chars.toCharArray())
    var ans = 0
    subStrings.forEach {
        var left = 0
        var right = it.length - 1
        while (left <= right) {
            if (chars.contains(it[left])) {
                left++
            } else break
            if (chars.contains(it[right])) {
                right--
            } else break
        }
        if (left <= right) ans = max(ans, longestSubstring(it.substring(left..right), k))
    }
    return ans
}

// 1763. 最长的美好子字符串
fun longestNiceSubstring(s: String): String {
    if (s.length <= 1) return ""
    val lowerCaseChars = mutableSetOf<Char>()
    val upperCaseChars = mutableSetOf<Char>()
    for (c in s) {
        if (c in 'a'..'z') lowerCaseChars.add(c)
        else if (c in 'A'..'Z') upperCaseChars.add(c.lowercaseChar())
    }
    if (lowerCaseChars == upperCaseChars) return s
    val chars =
        (lowerCaseChars subtract upperCaseChars) union (upperCaseChars subtract lowerCaseChars).map { it.uppercaseChar() }
    val subStrings = s.split(*chars.toCharArray())
    var ans = ""
    subStrings.forEach {
        val resStr = longestNiceSubstring(it)
        ans = if (resStr.length > ans.length) resStr else ans
    }
    return ans
}

// 2000. 反转单词前缀
fun reversePrefix(word: String, ch: Char): String {
    val stack = ArrayDeque<Char>()
    var idx = 0
    while (idx < word.length) {
        stack.addFirst(word[idx])
        if (word[idx] == ch) break
        idx++
    }
    if (idx == word.length) return word
    return stack.joinToString(separator = "", transform = { it.toString() }) + word.substring(idx + 1)
}

// 1414. 和为 K 的最少斐波那契数字数目
fun findMinFibonacciNumbers(k: Int): Int {
    val fibos = ArrayDeque(listOf(2, 1))
    while (fibos.first() < k) {
        val nxt = fibos[0] + fibos[1]
        if (nxt > k) break
        fibos.addFirst(nxt)
    }
    var ans = 0
    var remainK = k
    for (num in fibos) {
        while (num <= remainK) {
            remainK -= num
            ans++
        }
        if (remainK == 0) break
    }
    return ans
}

// 1748. 唯一元素的和
fun sumOfUnique(nums: IntArray): Int {
    val set = mutableSetOf<Int>()
    val multiSet = mutableSetOf<Int>()
    for (num in nums) {
        if (set.contains(num)) {
            set.remove(num)
            multiSet.add(num)
        } else if (!multiSet.contains(num)) {
            set.add(num)
        }
    }
    return set.sum()
}

// 1405. 最长快乐字符串
fun longestDiverseString_dfs(now: Pair<String, Int>, remains: PriorityQueue<Pair<String, Int>>): String {
    var ret = ""
    val nxt = remains.poll() ?: return ""
    val newNow: Pair<String, Int>
    if (now.second > 0) remains.add(now)
    if (nxt.second > now.second && nxt.second >= 2) {
        newNow = Pair(nxt.first, nxt.second - 2)
        ret += nxt.first.repeat(2)
        ret += longestDiverseString_dfs(newNow, remains)
    } else if (nxt.second >= 1) {
        newNow = Pair(nxt.first, nxt.second - 1)
        ret += nxt.first
        ret += longestDiverseString_dfs(newNow, remains)
    }
    return ret
}

fun longestDiverseString(a: Int, b: Int, c: Int): String {
    val comparator = compareByDescending<Pair<String, Int>> { it.second }
    val remains = PriorityQueue(comparator)
    if (a > 0) remains.add(Pair("a", a))
    if (b > 0) remains.add(Pair("b", b))
    if (c > 0) remains.add(Pair("c", c))
    val now = remains.poll()
    var nowStr: String
    val newNow: Pair<String, Int>
    if (now.second >= 2) {
        newNow = Pair(now.first, now.second - 2)
        nowStr = now.first.repeat(2)
    } else {
        newNow = Pair(now.first, 0)
        nowStr = now.first
    }
    return nowStr + longestDiverseString_dfs(newNow, remains)
}

// 1001. 网格照明
fun gridIllumination_getD1(cord: IntArray) =
    intArrayOf(cord[0] - Math.min(cord[0], cord[1]), cord[1] - Math.min(cord[0], cord[1])).contentToString()

fun gridIllumination_getD2(cord: IntArray, n: Int): String {
    val ret = if (cord[0] + cord[1] >= n) intArrayOf(cord[0] + cord[1] - n + 1, n - 1)
    else intArrayOf(0, cord[0] + cord[1])
    return ret.contentToString()
}

fun gridIllumination_closeOne(key: String, map: MutableMap<String, Int>) {
    val newVal = map.getOrDefault(key, 1) - 1
    if (newVal == 0) map.remove(key)
    else map[key] = newVal
}

fun gridIllumination_closeOne(key: Int, map: MutableMap<Int, Int>) {
    val newVal = map.getOrDefault(key, 1) - 1
    if (newVal == 0) map.remove(key)
    else map[key] = newVal
}

fun gridIllumination_close(
    cord: IntArray, n: Int, lampSet: MutableSet<String>,
    row: MutableMap<Int, Int>, col: MutableMap<Int, Int>,
    diag1: MutableMap<String, Int>, diag2: MutableMap<String, Int>
) {
    val nearBys = arrayOf(
        intArrayOf(-1, -1),
        intArrayOf(-1, 1),
        intArrayOf(1, 1),
        intArrayOf(1, -1),
        intArrayOf(0, 1),
        intArrayOf(0, -1),
        intArrayOf(1, 0),
        intArrayOf(-1, 0),
        intArrayOf(0, 0)
    )
    for (nearBy in nearBys) {
        val lampNearBy = intArrayOf(cord[0] + nearBy[0], cord[1] + nearBy[1])
        if (lampNearBy[0] < 0 || lampNearBy[1] < 0 || lampNearBy[0] >= n || lampNearBy[1] >= n) continue
        if (lampSet.contains(lampNearBy.contentToString())) {
            lampSet.remove(lampNearBy.contentToString())
            gridIllumination_closeOne(lampNearBy[0], row)
            gridIllumination_closeOne(lampNearBy[1], col)
            gridIllumination_closeOne(gridIllumination_getD1(lampNearBy), diag1)
            gridIllumination_closeOne(gridIllumination_getD2(lampNearBy, n), diag2)
        }
    }
}

fun gridIllumination(n: Int, lamps: Array<IntArray>, queries: Array<IntArray>): IntArray {
    val ans = mutableListOf<Int>()
    val lampSet = mutableSetOf<String>()
    val row = mutableMapOf<Int, Int>()
    val col = mutableMapOf<Int, Int>()
    val diag1 = mutableMapOf<String, Int>() // 左上右下
    val diag2 = mutableMapOf<String, Int>() // 右上左下
    for (lamp in lamps) {
        if (lampSet.contains(lamp.contentToString())) continue
        lampSet.add(lamp.contentToString())
        row[lamp[0]] = row.getOrDefault(lamp[0], 0) + 1
        col[lamp[1]] = col.getOrDefault(lamp[1], 0) + 1
        val d1 = gridIllumination_getD1(lamp)
        diag1[d1] = diag1.getOrDefault(d1, 0) + 1
        val d2 = gridIllumination_getD2(lamp, n)
        diag2[d2] = diag2.getOrDefault(d2, 0) + 1
    }
    for (query in queries) {
        if (row.containsKey(query[0])
            || col.containsKey(query[1])
            || diag1.containsKey(gridIllumination_getD1(query))
            || diag2.containsKey(gridIllumination_getD2(query, n))
        ) {
            ans.add(1)
        } else {
            ans.add(0)
        }
        gridIllumination_close(query, n, lampSet, row, col, diag1, diag2)
    }
    return ans.toIntArray()
}

// 2006. 差的绝对值为 K 的数对数目
fun countKDifference(nums: IntArray, k: Int): Int {
    val cnt = IntArray(101) { 0 }
    var ans = 0
    nums.forEach { cnt[it]++ }
    for (i in 1..100 - k) {
        ans += cnt[i] * cnt[i + k]
    }
    return ans
}

// 1447. 最简分数
fun simplifiedFractions_gcd(a: Int, b: Int): Int = if (b == 0) a else simplifiedFractions_gcd(b, a % b)
fun simplifiedFractions(n: Int): List<String> {
    val ans = mutableListOf<String>()
    for (i in 1..n) {
        for (j in 1 until i) {
            if (simplifiedFractions_gcd(i, j) == 1) {
                ans.add("$j/$i")
            }
        }
    }
    return ans
}

// 1984. 学生分数的最小差值
fun minimumDifference(nums: IntArray, k: Int): Int {
    val sortedNums = nums.sorted()
    var ans = sortedNums[k - 1] - sortedNums[0]
    for (i in 0 until nums.size - k + 1) {
        ans = min(sortedNums[i + k - 1] - sortedNums[i], ans)
    }
    return ans
}

// 1020. 飞地的数量
fun numEnclaves(grid: Array<IntArray>): Int {
    val nextSteps = arrayOf(intArrayOf(1, 0), intArrayOf(-1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
    val queue = ArrayDeque<IntArray>()
    val map = Array(grid.size) { intArrayOf(*grid[it]) }
    var ans = 0
    for (i in map.indices) {
        if (map[i][0] == 1) {
            map[i][0] = 2
            queue.addLast(intArrayOf(i, 0))
        }
        if (map[i][map[i].size - 1] == 1) {
            map[i][map[i].size - 1] = 2
            queue.addLast(intArrayOf(i, map[i].size - 1))
        }
    }
    for (j in map[0].indices) {
        if (map[0][j] == 1) {
            map[0][j] = 2
            queue.addLast(intArrayOf(0, j))
        }
        if (map[map.size - 1][j] == 1) {
            map[map.size - 1][j] = 2
            queue.addLast(intArrayOf(map.size - 1, j))
        }
    }
    while (queue.isNotEmpty()) {
        val now = queue.removeFirst()
        for (nextStep in nextSteps) {
            val next = intArrayOf(now[0] + nextStep[0], now[1] + nextStep[1])
            if (next[0] >= 0 && next[1] >= 0 && next[0] < map.size && next[1] < map[0].size) {
                if (map[next[0]][next[1]] == 1) {
                    map[next[0]][next[1]] = 2
                    queue.addLast(next)
                }
            }
        }
    }
    for (i in map.indices) {
        for (j in map[0].indices) {
            if (map[i][j] == 1) ans++
        }
    }
    return ans
}

// 28. 实现 strStr()
fun strStr(haystack: String, needle: String): Int {
    if (needle.isEmpty()) return 0
    var pStr = 0
    while (pStr < haystack.length) {
        if (haystack[pStr] == needle[0]) {
            var pSubStr = 0
            var pTmpStr = pStr
            while (pSubStr < needle.length && pTmpStr < haystack.length) {
                if (haystack[pTmpStr] == needle[pSubStr]) {
                    if (pSubStr == needle.length - 1) return pStr
                    pSubStr++
                    pTmpStr++
                } else {
                    break
                }
            }
        }
        pStr++
    }
    return -1
}

// 236. 二叉树的最近公共祖先
fun lowestCommonAncestor(root: TreeNode?, p: TreeNode?, q: TreeNode?): TreeNode? {
    if (root == null || p == null || q == null) return null
    val parents = mutableMapOf<Int, TreeNode>()
    val queue = ArrayDeque<TreeNode>()
    queue.addLast(root)
    while (queue.isNotEmpty()) {
        val now = queue.removeFirst()
        now.left?.let {
            parents[it.`val`] = now
            queue.addLast(it)
        }
        now.right?.let {
            parents[it.`val`] = now
            queue.addLast(it)
        }
    }
    val ancestors = mutableSetOf<Int>()
    var p1 = p.`val`
    ancestors.add(p1)
    while (parents[p1] != null) {
        ancestors.add(parents[p1]!!.`val`)
        p1 = parents[p1]!!.`val`
    }
    var ans: TreeNode = q
    while (!ancestors.contains(ans.`val`)) {
        ans = parents[ans.`val`]!!
    }
    return ans
}

// 540. 有序数组中的单一元素
fun singleNonDuplicate(nums: IntArray): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = (left + right) / 2
        if (mid == left) return nums[mid]
        if (mid % 2 == 0) {
            if (nums[mid] == nums[mid + 1]) {
                left = mid
            } else {
                right = mid
            }
        } else {
            if (nums[mid] == nums[mid + 1]) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
    }
    return -1
}

fun main() {
    val num = 3
    val IntArray2d = arrayOf(
        intArrayOf(0, 0),
        intArrayOf(0, 4)
    )
    val intArray = intArrayOf(1,1,2)
    val ans = singleNonDuplicate(intArray)
    println(ans)
}