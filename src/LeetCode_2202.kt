import java.util.PriorityQueue
import kotlin.collections.ArrayDeque
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random
import kotlin.random.nextInt

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

// 1380. 矩阵中的幸运数
fun luckyNumbers(matrix: Array<IntArray>): List<Int> {
    val cols = matrix[0].size
    val maxNumIdxInCols = MutableList(cols) { 0 }
    for (j in 0 until cols) {
        for (i in matrix.indices) {
            if (matrix[i][j] > matrix[maxNumIdxInCols[j]][j]) maxNumIdxInCols[j] = i
        }
    }
    val ans = mutableListOf<Int>()
    for (i in matrix.indices) {
        val row = matrix[i]
        val minNumColIdx = row.foldIndexed(0) { idx, acc, num -> if (num < row[acc]) idx else acc }
        if (maxNumIdxInCols[minNumColIdx] == i) ans.add(row[minNumColIdx])
    }
    return ans
}

// 144. 二叉树的前序遍历
fun preorderTraversal(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    val ans = mutableListOf<Int>()
    val stack = ArrayDeque<TreeNode>()
    stack.addLast(root)
    while (stack.isNotEmpty()) {
        val node = stack.removeLast()
        ans.add(node.`val`)
        node.right?.let {
            stack.addLast(it)
        }
        node.left?.let {
            stack.addLast(it)
        }
    }
    return ans
}

// 94. 二叉树的中序遍历
fun inorderTraversal(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    val ans = mutableListOf<Int>()
    val stack = ArrayDeque<TreeNode>()
    var node = root
    while (node != null || stack.isNotEmpty()) {
        while (node != null) {
            stack.addLast(node)
            node = node.left
        }
        node = stack.removeLast()
        ans.add(node.`val`)
        if (node.right == null) {
            // 没有右子树，node 遍历完毕
            node = null
        } else {
            node = node.right
        }
    }
    return ans
}

fun inorderTraversalSolution2(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    val ans = mutableListOf<Int>()
    val stack = ArrayDeque<Pair<TreeNode, Int>>() // 0 没遍历过
    stack.addLast(Pair(root, 0))
    while (stack.isNotEmpty()) {
        val pair = stack.removeLast()
        if (pair.second == 0) { // 没遍历过
            pair.first.right?.let { stack.addLast(Pair(it, 0)) }
            stack.addLast(Pair(pair.first, 1))
            pair.first.left?.let { stack.addLast(Pair(it, 0)) }
        } else {
            ans.add(pair.first.`val`)
        }
    }
    return ans
}

// 145. 二叉树的后序遍历
fun postorderTraversal(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    var node = root
    var pre: TreeNode? = null
    val stack = ArrayDeque<TreeNode>()
    val ans = mutableListOf<Int>()
    // node 表示的是下一个要遍历的节点，
    // node 不为空表示它的左子树和右子树都没有遍历，
    // node 为空表示现在是在往回返
    while (node != null || stack.isNotEmpty()) {
        // 左右子树都没遍历过的通通先找最左
        while (node != null) {
            stack.addLast(node)
            node = node.left
        }
        node = stack.removeLast()
        // stack 弹出的已经没有了左子树，所以看它的右子树
        // 要么右子树为空，要么右子树遍历过
        if (node.right == null || pre == node.right) {
            ans.add(node.`val`)
            pre = node
            node = null
        } else {
            // 有右子树且没被遍历，那 node 还得进栈待着，先遍历右子树
            stack.addLast(node)
            node = node.right
        }
    }
    return ans
}

// 92. 反转链表 II
fun reverseBetween(head: ListNode?, left: Int, right: Int): ListNode? {
    if (head == null) return null
    val stack = ArrayDeque<ListNode>()
    val dummyHead = ListNode(-1)
    dummyHead.next = head
    var node = head
    var start = dummyHead
    repeat(left - 1) {
        start = node!!
        node = node!!.next
    }
    repeat(right - left + 1) {
        stack.addLast(node!!)
        node = node!!.next
    }
    while (stack.isNotEmpty()) {
        start.next = stack.removeLast()
        start = start.next!!
    }
    start.next = node
    return dummyHead.next
}

fun reverseBetweenSolution2(head: ListNode?, left: Int, right: Int): ListNode? {
    if (head == null) return null
    val dummyHead = ListNode(-1)
    dummyHead.next = head
    var front = dummyHead
    repeat(left - 1) {
        front = front.next!!
    }
    var realHead = front.next!!
    var next: ListNode? = realHead.next
    repeat(right - left) {
        next = next!!.next
        realHead.next!!.next = realHead
        realHead = realHead.next!!
    }
    return dummyHead.next
}

// 138. 复制带随机指针的链表
class Node(var `val`: Int) {
    var next: Node? = null
    var random: Node? = null
}

fun copyRandomList(node: Node?): Node? {
    if (node == null) return null
    val map = mutableMapOf<Node, Node>()
    var oldNode: Node = node
    val newHead = Node(oldNode.`val`)
    var newNode = newHead

    map[oldNode] = newNode

    while (oldNode.next != null) {
        val newNext = map.getOrDefault(oldNode.next!!, Node(oldNode.next!!.`val`))
        newNode.next = newNext
        map[oldNode.next!!] = newNext

        oldNode.random?.let {
            val newRandom = map.getOrDefault(it, Node(it.`val`))
            newNode.random = newRandom
            map[it] = newRandom
        }
        newNode = newNode.next!!
        oldNode = oldNode.next!!
    }

    oldNode.random?.let {
        newNode.random = map[it]
    }
    return newHead
}

// 232. 用栈实现队列
class MyQueue() {
    private val inStack = ArrayDeque<Int>()
    private val outStack = ArrayDeque<Int>()
    fun push(x: Int) {
        inStack.addLast(x)
    }

    private fun ensure() {
        if (outStack.isNotEmpty()) return
        while (inStack.isNotEmpty()) {
            outStack.addLast(inStack.removeLast())
        }
    }

    fun pop(): Int {
        ensure()
        return outStack.removeLast()
    }

    fun peek(): Int {
        ensure()
        return outStack.last()
    }

    fun empty(): Boolean {
        return inStack.isEmpty() && outStack.isEmpty()
    }
}

// 剑指 Offer 38. 字符串的排列
fun permutation_swap(s: CharArray, idx1: Int, idx2: Int) {
    val tmp = s[idx1]
    s[idx1] = s[idx2]
    s[idx2] = tmp
}

fun permutation_dfs(s: CharArray, startIdx: Int, ans: MutableList<String>) {
    if (startIdx == s.size - 1) {
        ans.add(String(s))
        return
    }
    val set = mutableSetOf<Char>()
    for (idx in startIdx until s.size) {
        if (set.contains(s[idx])) continue
        set.add(s[idx])
        permutation_swap(s, idx, startIdx)
        permutation_dfs(s, startIdx + 1, ans)
        permutation_swap(s, idx, startIdx)
    }
}

fun permutation(s: String): Array<String> {
    val ans = mutableListOf<String>()
    permutation_dfs(s.toCharArray(), 0, ans)
    return ans.toTypedArray()
}

// 567. 字符串的排列
fun checkInclusion(s1: String, s2: String): Boolean {
    val set1 = IntArray(26) { 0 }
    val set2 = IntArray(26) { 0 }
    if (s1.length > s2.length) return false
    for (idx in s1.indices) {
        set1[s1[idx] - 'a'] += 1
        set2[s2[idx] - 'a'] += 1
    }
    if (set1.contentEquals(set2)) return true
    for (idx in s1.length until s2.length) {
        set2[s2[idx] - 'a'] += 1
        set2[s2[idx - s1.length] - 'a'] -= 1
        if (set1.contentEquals(set2)) return true
    }
    return false
}

// 76. 最小覆盖子串
fun minWindow(s: String, t: String): String {
    val cntS = mutableMapOf<Char, Int>()
    val cntT = mutableMapOf<Char, Int>()
    t.forEach { cntT[it] = cntT.getOrDefault(it, 0) + 1 }
    var minLen = Int.MAX_VALUE
    var right = 0
    var left = 0
    var match = 0
    var start = 0
    while (right < s.length) {
        cntS[s[right]] = cntS.getOrDefault(s[right], 0) + 1
        if (cntT.containsKey(s[right]) && cntS[s[right]] == cntT[s[right]]) {
            match++
        }
        while (match == cntT.size) {
            if (minLen > right - left + 1) {
                start = left
                minLen = right - left + 1
            }
            cntS[s[left]] = cntS.getOrDefault(s[left], 1) - 1
            if (cntS.getOrDefault(s[left], 0) < cntT.getOrDefault(s[left], 0)) match--
            left++
        }
        right++
    }
    return if (minLen == Int.MAX_VALUE) "" else s.substring(start, start + minLen)
}

// 98. 验证二叉搜索树
fun isValidBST_dfs(root: TreeNode?, minVal: Long, maxVal: Long): Boolean {
    if (root == null) return true
    if (root.`val` >= maxVal || root.`val` <= minVal) return false
    return isValidBST_dfs(root.left, minVal, root.`val`.toLong()) && isValidBST_dfs(
        root.right,
        root.`val`.toLong(),
        maxVal
    )
}

fun isValidBST(root: TreeNode?): Boolean {
    return isValidBST_dfs(root, Long.MIN_VALUE, Long.MAX_VALUE)
}

// 93. 复原 IP 地址
fun restoreIpAddresses_isValidIP(s: String): Boolean {
    if (s.isEmpty()) return false
    if (s[0] == '0' && s.length > 1) return false
    val num = s.toIntOrNull()
    if (num == null || num > 255) return false
    return true
}

fun restoreIpAddresses_dfs(s: String, nowState: MutableList<String>, nowIdx: Int, ans: MutableSet<String>) {
    if (nowState.size == 3) {
        val ip4 = s.substring(nowIdx)
        if (restoreIpAddresses_isValidIP(ip4)) {
            val ipAddr = nowState.joinToString(".") + "." + ip4
            ans.add(ipAddr)
        }
        return
    }
    for (endIdx in nowIdx until Math.min(nowIdx + 3, s.length - 3 + nowState.size)) {
        val ipn = s.substring(nowIdx, endIdx + 1)
        if (restoreIpAddresses_isValidIP(ipn)) {
            nowState.add(ipn)
            restoreIpAddresses_dfs(s, nowState, endIdx + 1, ans)
            nowState.removeAt(nowState.size - 1)
        }
    }
}

fun restoreIpAddresses(s: String): List<String> {
    if (s.length > 12) return emptyList()
    val ans = mutableSetOf<String>()
    val state = mutableListOf<String>()
    restoreIpAddresses_dfs(s, state, 0, ans)
    return ans.toList()
}

// 22. 括号生成
fun generateParenthesis_dfs(now: String, left: Int, right: Int, ans: MutableList<String>) {
    if (left == 0) {
        val s = now + ")".repeat(right)
        ans.add(s)
        return
    }
    if (left < right) {
        generateParenthesis_dfs(now + ")", left, right - 1, ans)
    }
    generateParenthesis_dfs(now + "(", left - 1, right, ans)
}

fun generateParenthesis(n: Int): List<String> {
    val ans = mutableListOf<String>()
    generateParenthesis_dfs("", n, n, ans)
    return ans
}

// 207. 课程表
fun canFinish(numCourses: Int, prerequisites: Array<IntArray>): Boolean {
    val adjs = Array<MutableList<Int>>(numCourses) { mutableListOf() }
    val queue = ArrayDeque<Int>()
    val courses = IntArray(numCourses) { 0 }
    var cnt = 0
    for (require in prerequisites) {
        adjs[require[1]].add(require[0])
        courses[require[0]]++
    }
    for (id in courses.indices) {
        if (courses[id] == 0) {
            queue.addLast(id)
            cnt++
        }
    }
    while (queue.isNotEmpty()) {
        for (nextCourse in adjs[queue.removeFirst()]) {
            courses[nextCourse]--
            if (courses[nextCourse] == 0) {
                queue.addLast(nextCourse)
                cnt++
            }
        }
    }
    return cnt == numCourses
}

// 210. 课程表 II
fun findOrder(numCourses: Int, prerequisites: Array<IntArray>): IntArray {
    val adjs = Array<MutableList<Int>>(numCourses) { mutableListOf() }
    val queue = ArrayDeque<Int>()
    val courses = IntArray(numCourses) { 0 }
    val ans = mutableListOf<Int>()
    var cnt = 0
    for (require in prerequisites) {
        adjs[require[1]].add(require[0])
        courses[require[0]]++
    }
    for (id in courses.indices) {
        if (courses[id] == 0) {
            queue.addLast(id)
            ans.add(id)
            cnt++
        }
    }
    while (queue.isNotEmpty()) {
        for (nextCourse in adjs[queue.removeFirst()]) {
            courses[nextCourse]--
            if (courses[nextCourse] == 0) {
                queue.addLast(nextCourse)
                ans.add(nextCourse)
                cnt++
            }
        }
    }
    return if (cnt != numCourses) intArrayOf() else ans.toIntArray()
}

// 662. 二叉树最大宽度
fun widthOfBinaryTree(root: TreeNode?): Int {
    if (root == null) return 0
    val queue = ArrayDeque<Pair<TreeNode, Int>>()
    queue.addLast(Pair(root, 1))
    var ans = 1
    while (queue.isNotEmpty()) {
        val leftIdx = queue.first().second
        val size = queue.size
        repeat(size) {
            val pair = queue.removeFirst()
            pair.first.left?.let { queue.addLast(Pair(it, pair.second * 2)) }
            pair.first.right?.let { queue.addLast(Pair(it, pair.second * 2 + 1)) }
            if (it == size - 1) {
                val rightIdx = pair.second
                ans = Math.max(rightIdx - leftIdx + 1, ans)
            }
        }
    }
    return ans
}

// 6. Z 字形变换
fun convert(s: String, numRows: Int): String {
    if (numRows == 1) return s
    val group = s.length / (2 * numRows - 2) + if (s.length % (2 * numRows - 2) == 0) 0 else 1
    val groupLen = numRows - 1
    val array2d = Array(numRows) { CharArray(group * groupLen) { ' ' } }
    var idxStr = 0
    for (groupIdx in 0 until group) {
        val startColIdx = groupIdx * groupLen
        for (i in 0 until numRows) {
            array2d[i][startColIdx] = s[idxStr]
            idxStr++
            if (idxStr >= s.length) break
        }
        if (idxStr >= s.length) break
        var colIdx = startColIdx + 1
        for (i in numRows - 2 downTo 1) {
            array2d[i][colIdx] = s[idxStr]
            idxStr++
            colIdx++
            if (idxStr >= s.length) break
        }
        if (idxStr >= s.length) break
    }
    val ans = CharArray(s.length) { ' ' }
    idxStr = 0
    for (i in array2d.indices) {
        for (j in array2d[i].indices) {
            if (array2d[i][j] == ' ') continue
            ans[idxStr] = array2d[i][j]
            idxStr++
        }
    }
    return ans.joinToString("")
}

// 199. 二叉树的右视图
fun rightSideView(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    val queue = ArrayDeque<TreeNode>()
    val ans = mutableListOf<Int>()
    queue.addLast(root)
    while (queue.isNotEmpty()) {
        val size = queue.size
        repeat(size) { idx ->
            val node = queue.removeFirst()
            node.left?.let { queue.addLast(it) }
            node.right?.let { queue.addLast(it) }
            if (idx == size - 1) {
                ans.add(node.`val`)
            }
        }
    }
    return ans
}

// 103. 二叉树的锯齿形层序遍历
fun zigzagLevelOrder(root: TreeNode?): List<List<Int>> {
    if (root == null) return emptyList()
    val ans = mutableListOf<MutableList<Int>>()
    val queue = ArrayDeque<TreeNode>()
    var leftFirst = true
    queue.addLast(root)
    while (queue.isNotEmpty()) {
        val stack = ArrayDeque<TreeNode>()
        val subAns = mutableListOf<Int>()
        repeat(queue.size) {
            val node = queue.removeFirst()
            subAns.add(node.`val`)
            if (leftFirst) {
                node.left?.let { stack.addLast(it) }
                node.right?.let { stack.addLast(it) }
            } else {
                node.right?.let { stack.addLast(it) }
                node.left?.let { stack.addLast(it) }
            }
        }
        ans.add(subAns)
        while (stack.isNotEmpty()) {
            queue.addLast(stack.removeLast())
        }
        leftFirst = !leftFirst
    }
    return ans
}

// 剑指 Offer 47. 礼物的最大价值
fun maxValue(grid: Array<IntArray>): Int {
    val row = grid.size
    val col = grid[0].size
    val dp = Array(row) { IntArray(col) { 0 } }
    dp[0][0] = grid[0][0]
    for (i in 1 until row) {
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    }
    for (i in 1 until col) {
        dp[0][i] = dp[0][i - 1] + grid[0][i]
    }
    for (i in 1 until row) {
        for (j in 1 until col) {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        }
    }
    return dp[row - 1][col - 1]
}

// 695. 岛屿的最大面积
var maxAreaOfIsland_ans = 0
fun maxAreaOfIsland_dfs(grid: Array<IntArray>, nowCord: IntArray): Int {
    val nexts = arrayOf(intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(-1, 0), intArrayOf(0, -1))
    var ans = 1
    for (next in nexts) {
        val nextCord = intArrayOf(nowCord[0] + next[0], nowCord[1] + next[1])
        if (nextCord[0] >= 0 && nextCord[1] >= 0 && nextCord[0] < grid.size && nextCord[1] < grid[0].size && grid[nextCord[0]][nextCord[1]] == 1) {
            grid[nextCord[0]][nextCord[1]] = 0
            ans += maxAreaOfIsland_dfs(grid, nextCord)
        }
    }
    return ans
}

fun maxAreaOfIsland(grid: Array<IntArray>): Int {
    for (i in grid.indices) {
        for (j in grid[i].indices) {
            if (grid[i][j] == 1) {
                grid[i][j] = 0
                maxAreaOfIsland_ans = Math.max(maxAreaOfIsland_ans, maxAreaOfIsland_dfs(grid, intArrayOf(i, j)))
            }
        }
    }
    return maxAreaOfIsland_ans
}

// 54. 螺旋矩阵
fun spiralOrder(matrix: Array<IntArray>): List<Int> {
    var top = 0
    var bottom = matrix.size - 1
    var left = 0
    var right = matrix[0].size - 1
    val ans = mutableListOf<Int>()
    var dir = 0
    while (top <= bottom && left <= right) {
        when (dir % 4) {
            0 -> {
                for (col in left..right) ans.add(matrix[top][col])
                top++
            }
            1 -> {
                for (row in top..bottom) ans.add(matrix[row][right])
                right--
            }
            2 -> {
                for (col in right downTo left) ans.add(matrix[bottom][col])
                bottom--
            }
            3 -> {
                for (row in bottom downTo top) ans.add(matrix[row][left])
                left++
            }
        }
        dir++
    }
    return ans
}

// 151. 翻转字符串里的单词
fun reverseWords(s: String): String {
    val s = buildString {
        var left = s.length - 1
        var right = s.length - 1
        while (left >= -1) {
            val digit = if (left == -1) ' ' else s[left]
            val lastDigit = if (left == s.length - 1) s[s.length - 1] else ' '
            if (digit != ' ' && lastDigit == ' ') {
                right = left
            } else if (digit == ' ' && lastDigit != ' ') {
                append(s.substring(left - 1..right))
                append(" ")
            }
            left--
        }
    }
    return s.substring(0 until s.length - 1)
}

// 3. 无重复字符的最长子串
fun lengthOfLongestSubstring(s: String): Int {
    var left = 0
    var right = 0
    var ans = 0
    val set = mutableSetOf<Char>()
    while (right < s.length) {
        val rightChar = s[right]
        if (set.contains(rightChar)) {
            while (s[left] != rightChar) {
                set.remove(s[left])
                left++
            }
            left++
            right++

        } else {
            set.add(rightChar)
            ans = max(ans, right - left + 1)
            right++
        }
    }
    return ans
}

// 56. 合并区间
fun merge(intervals: Array<IntArray>): Array<IntArray> {
    val sortedIntervals = intervals.sortedBy { it[0] }
    val ans = mutableListOf<IntArray>()
    ans.add(sortedIntervals[0])
    for (interval in sortedIntervals) {
        val start = interval[0]
        val end = interval[1]
        if (start > ans.last()[1]) {
            ans.add(interval)
        } else {
            val lastInterval = ans.removeLast()
            lastInterval[1] = Math.max(lastInterval[1], end)
            ans.add(lastInterval)
        }
    }
    return ans.toTypedArray()
}

// 105. 从前序与中序遍历序列构造二叉树
fun buildTree(preorder: IntArray, inorder: IntArray): TreeNode? {
    if (preorder.isEmpty() && inorder.isEmpty()) return null
    val rootVal = preorder[0]
    val root = TreeNode(rootVal)
    for (i in inorder.indices) {
        if (inorder[i] == rootVal) {
            root.left = buildTree(preorder.sliceArray(1 until i + 1), inorder.sliceArray(0 until i))
            root.right =
                buildTree(preorder.sliceArray(i + 1 until preorder.size), inorder.sliceArray(i + 1 until inorder.size))
        }
    }
    return root
}

// 53. 最大子数组和
fun maxSubArray(nums: IntArray): Int {
    val dp = IntArray(nums.size) { nums[it] }
    var ans = dp[0]
    for (idx in 1 until dp.size) {
        dp[idx] = Math.max(dp[idx], dp[idx] + dp[idx - 1])
        ans = Math.max(dp[idx], ans)
    }
    return ans
}

// 112. 路径总和
fun hasPathSum_dfs(root: TreeNode, targetSum: Int, nowSum: Int): Boolean {
    if (root.left == null && root.right == null) {
        if (nowSum + root.`val` == targetSum) return true
        return false
    }
    var ret = false
    root.left?.let {
        ret = ret || hasPathSum_dfs(it, targetSum, nowSum + root.`val`)
    }
    root.right?.let {
        ret = ret || hasPathSum_dfs(it, targetSum, nowSum + root.`val`)
    }
    return ret
}

fun hasPathSum(root: TreeNode?, targetSum: Int): Boolean {
    if (root == null) return false
    return hasPathSum_dfs(root, targetSum, 0)
}

// 113. 路径总和 II
fun pathSum_dfs(root: TreeNode, targetSum: Int, nowSum: Int, nowPath: MutableList<Int>, ans: MutableList<List<Int>>) {
    if (root.left == null && root.right == null) {
        if (nowSum + root.`val` == targetSum) {
            nowPath.add(root.`val`)
            ans.add(nowPath.toList())
            nowPath.removeAt(nowPath.size - 1)
        }
        return
    }

    root.left?.let {
        nowPath.add(root.`val`)
        pathSum_dfs(it, targetSum, nowSum + root.`val`, nowPath, ans)
        nowPath.removeAt(nowPath.size - 1)
    }
    root.right?.let {
        nowPath.add(root.`val`)
        pathSum_dfs(it, targetSum, nowSum + root.`val`, nowPath, ans)
        nowPath.removeAt(nowPath.size - 1)
    }
}

fun pathSum(root: TreeNode?, targetSum: Int): List<List<Int>> {
    if (root == null) return emptyList()
    val ans = mutableListOf<List<Int>>()
    val nowPath = mutableListOf<Int>()
    pathSum_dfs(root, targetSum, 0, nowPath, ans)
    return ans
}

// 79. 单词搜索
fun exist_dfs(
    board: Array<CharArray>,
    vis: Array<BooleanArray>,
    nowIdx: IntArray,
    word: String,
    wordIdx: Int
): Boolean {
    if (wordIdx == word.length) return true
    val nextSteps = arrayOf(intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0))
    for (nextStep in nextSteps) {
        val nextIdx = intArrayOf(nowIdx[0] + nextStep[0], nowIdx[1] + nextStep[1])
        if (nextIdx[0] >= 0 && nextIdx[1] >= 0 && nextIdx[0] < board.size && nextIdx[1] < board[0].size && board[nextIdx[0]][nextIdx[1]] == word[wordIdx] && !vis[nextIdx[0]][nextIdx[1]]) {
            vis[nextIdx[0]][nextIdx[1]] = true
            val ret = exist_dfs(board, vis, nextIdx, word, wordIdx + 1)
            vis[nextIdx[0]][nextIdx[1]] = false
            if (ret) return true
        }
    }
    return false
}

fun exist(board: Array<CharArray>, word: String): Boolean {
    for (i in board.indices) {
        for (j in board[i].indices) {
            if (board[i][j] == word[0]) {
                val vis = Array(board.size) { BooleanArray(board[0].size) { false } }
                vis[i][j] = true
                val ret = exist_dfs(board, vis, intArrayOf(i, j), word, 1)
                if (ret) return true
            }
        }
    }
    return false
}

// 124. 二叉树中的最大路径和
var maxPathSum_sum: Long = Int.MIN_VALUE.toLong()
fun maxPathSum_dfs(root: TreeNode?): Long {
    if (root == null) return Int.MIN_VALUE.toLong()
    var ret = root.`val`.toLong()

    val leftVal = maxPathSum_dfs(root.left)
    ret = Math.max(ret, root.`val` + leftVal)
    maxPathSum_sum = Math.max(leftVal, maxPathSum_sum)
    val rightVal = maxPathSum_dfs(root.right)
    ret = Math.max(ret, root.`val` + rightVal)
    maxPathSum_sum = Math.max(leftVal, maxPathSum_sum)
    maxPathSum_sum = Math.max(rightVal, maxPathSum_sum)
    maxPathSum_sum = Math.max((leftVal + rightVal + root.`val`), maxPathSum_sum)
    return ret
}

fun maxPathSum(root: TreeNode?): Int {
    val sum = maxPathSum_dfs(root)
    return Math.max(sum, maxPathSum_sum).toInt()
}

// 347. 前 K 个高频元素
fun topKFrequent_qSort(nums: IntArray, left: Int, right: Int, map: MutableMap<Int, Int>): Int {
    findKthLargest_swap(nums, left, (left..right).random())
    val chosen = map[nums[left]]!!
    var pos = left
    for (idx in left + 1..right) {
        if (chosen > map[nums[idx]]!!) {
            pos++
            findKthLargest_swap(nums, pos, idx)
        }
    }
    findKthLargest_swap(nums, left, pos)
    return pos
}

fun topKFrequent(nums: IntArray, k: Int): IntArray {
    val cntMap = mutableMapOf<Int, Int>()
    for (num in nums) {
        cntMap[num] = cntMap.getOrDefault(num, 0) + 1
    }

    val realNums = IntArray(cntMap.size)
    for ((idx, key) in cntMap.keys.withIndex()) {
        realNums[idx] = key
    }
    val realK = realNums.size - k
    var left = 0
    var right = realNums.size - 1
    while (left <= right) {
        val mid = topKFrequent_qSort(realNums, left, right, cntMap)
        if (mid == realK) {
            break
        } else if (realK > mid) {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    val ans = mutableListOf<Int>()
    for (idx in realK until realNums.size) {
        ans.add(realNums[idx])
    }
    return ans.toIntArray()
}

// 162. 寻找峰值
fun findPeakElement_get(nums: IntArray, idx: Int): Long {
    if (idx in nums.indices) {
        return nums[idx].toLong()
    }
    return Long.MIN_VALUE
}

fun findPeakElement(nums: IntArray): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = (left + right) / 2
        val midVal = nums[mid]
        if (midVal > findPeakElement_get(nums, mid - 1) && midVal > findPeakElement_get(nums, mid + 1)) {
            return mid
        } else if (midVal < findPeakElement_get(nums, mid + 1)) {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}

// 155. 最小栈
class MinStack() {
    private val stack = ArrayDeque<Int>()
    private val minStack = ArrayDeque<Int>()

    fun push(`val`: Int) {
        stack.addLast(`val`)
        minStack.addLast(if (minStack.isNotEmpty()) Math.min(`val`, minStack.last()) else `val`)
    }

    fun pop() {
        stack.removeLast()
        minStack.removeLast()
    }

    fun top(): Int {
        return stack.last()
    }

    fun getMin(): Int {
        return minStack.last()
    }

}

// 215. 数组中的第K个最大元素
fun findKthLargest_swap(nums: IntArray, idx1: Int, idx2: Int) {
    val tmp = nums[idx1]
    nums[idx1] = nums[idx2]
    nums[idx2] = tmp
}

fun findKthLargest_sortK(nums: IntArray, left: Int, right: Int): Int {
    findKthLargest_swap(nums, left, Random.nextInt(left..right))
    val chosen = nums[left]
    var pos = left
    for (idx in left + 1..right) {
        if (nums[idx] < chosen) {
            pos++
            findKthLargest_swap(nums, pos, idx)
        }
    }
    findKthLargest_swap(nums, pos, left)
    return pos
}

fun findKthLargest(nums: IntArray, k: Int): Int {
    val target = nums.size - k
    var left = 0
    var right = nums.size - 1
    var idxK = findKthLargest_sortK(nums, left, right)
    while (idxK != target) {
        if (idxK > target) {
            right = idxK - 1
        } else {
            left = idxK + 1
        }
        idxK = findKthLargest_sortK(nums, left, right)
    }
    return nums[idxK]
}

// 206. 反转链表
fun reverseList(head: ListNode?): ListNode? {
    if (head == null) return null
    var node = head.next
    var ansHead = head
    ansHead.next = null
    while (node != null) {
        val next = node.next
        node.next = ansHead
        ansHead = node
        node = next
    }
    return ansHead
}

// 25. K 个一组翻转链表
fun reverseKGroup(head: ListNode?, k: Int): ListNode? {
    val reversedFakeNode = ListNode(-1)
    var lastGroupTail = reversedFakeNode // 上一组的末尾
    var tmpNode = head
    while (tmpNode != null) {
        val groupHead = tmpNode
        repeat(k - 1) {
            tmpNode = tmpNode!!.next
            if (tmpNode == null) {
                lastGroupTail.next = groupHead
                return reversedFakeNode.next
            }
        }
        val tail = tmpNode
        tmpNode = tmpNode!!.next
        tail!!.next = null
        val reversedGroupHead = reverseList(groupHead)
        lastGroupTail.next = reversedGroupHead
        lastGroupTail = groupHead
    }
    return reversedFakeNode.next
}

// 82. 删除排序链表中的重复元素 II
fun deleteDuplicates(head: ListNode?): ListNode? {
    if (head == null) return null
    val fakeNode = ListNode(-1)
    var tail = fakeNode
    var node: ListNode = head
    var duplicated = false
    while (node.next != null) {
        if (node.`val` == node.next!!.`val`) {
            duplicated = true
            node = node.next!!
            continue
        }
        if (duplicated) {
            duplicated = false
            node = node.next!!
            continue
        }
        tail.next = node
        tail = node
        node = node.next!!
    }
    if (!duplicated) {
        tail.next = node
    } else {
        tail.next = null
    }
    return fakeNode.next
}

// 81. 搜索旋转排序数组 II
fun search_2(nums: IntArray, target: Int): Boolean {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = (left + right) / 2
        val midVal = nums[mid]
        val leftVal = nums[left]
        val rightVal = nums[right]
        if (midVal == target) return true
        if (midVal < leftVal) {
            if (target in (midVal + 1)..rightVal) left = mid + 1
            else right = mid - 1
        } else if (midVal == leftVal) {
            left += 1
        } else {
            if (target in leftVal until midVal) right = mid - 1
            else left = mid + 1
        }
    }
    return false
}

// 33. 搜索旋转排序数组
fun search_1(nums: IntArray, target: Int): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = (left + right) / 2
        val midVal = nums[mid]
        val leftVal = nums[left]
        val rightVal = nums[right]
        if (midVal == target) return mid
        else if (midVal > rightVal) {
            if (target > midVal || target <= rightVal) left = mid + 1
            else right = mid - 1
        } else if (midVal < leftVal) {
            if (target >= leftVal || target < midVal) right = mid - 1
            else left = mid + 1
        } else {
            if (target > midVal) left = mid + 1
            else right = mid - 1
        }
    }
    return -1
}

fun search_1_Solution2(nums: IntArray, target: Int): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = (left + right) / 2
        val midVal = nums[mid]
        val leftVal = nums[left]
        val rightVal = nums[right]
        if (midVal == target) return mid
        if (midVal < leftVal) {
            if (target in (midVal + 1)..rightVal) left = mid + 1
            else right = mid - 1
        } else {
            if (target in leftVal until midVal) right = mid - 1
            else left = mid + 1
        }
    }
    return -1
}

// 143. 重排链表
fun reorderList(head: ListNode?) {
    var node = head!!
    val stack = ArrayDeque<ListNode>()

    var tmpNode = head
    while (tmpNode != null) {
        stack.addLast(tmpNode)
        tmpNode = tmpNode.next
    }

    while (node.next != null) {
        val next = node.next!!
        val reversedNext = stack.removeLast()
        node.next = reversedNext
        reversedNext.next = next
        if (next.next == reversedNext) {
            next.next = null
            return
        }
        node = next
    }
}

// 143. 重排链表
fun toMiddle(head: ListNode): ListNode {
    var slowP: ListNode = head
    var fastP: ListNode? = head
    var pre = head
    while (fastP != null && fastP.next != null) {
        pre = slowP
        slowP = slowP.next!!
        fastP = fastP.next!!.next
    }
    if (fastP == null) {
        pre.next = null
        return slowP
    }
    pre = slowP
    slowP = slowP.next!!
    pre.next = null
    return slowP
}

fun reverseList1(head: ListNode): ListNode {
    var reversedHead: ListNode = head
    var node = head.next
    head.next = null
    while (node != null) {
        val nextNode = node.next
        node.next = reversedHead
        reversedHead = node
        node = nextNode
    }
    return reversedHead
}

fun reorderListSolution1(head: ListNode?): Unit {
    if (head?.next == null) return
    var node1: ListNode? = head
    var node2: ListNode? = reverseList1(toMiddle(head))
    while (node1 != null && node2 != null) {
        val node1Next = node1.next
        val node2Next = node2.next
        node1.next = node2
        node2.next = node1Next
        node1 = node1Next
        node2 = node2Next
    }
}

// 21. 合并两个有序链表
fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
    val fakeNode = ListNode(-1)
    var mergedNodeTail = fakeNode
    var node1 = list1
    var node2 = list2
    while (node1 != null && node2 != null) {
        if (node1.`val` > node2.`val`) {
            mergedNodeTail.next = node2
            node2 = node2.next
        } else {
            mergedNodeTail.next = node1
            node1 = node1.next
        }
        mergedNodeTail = mergedNodeTail.next!!
    }
    mergedNodeTail.next = node1 ?: node2
    return fakeNode.next
}

// 148. 排序链表
fun sortList(head: ListNode?): ListNode? {
    if (head?.next == null) return head
    var slow: ListNode = head
    var fast = head
    var pre = head
    while (fast?.next != null) {
        pre = slow
        slow = slow.next!!
        fast = fast.next!!.next
    }
    val head2 = slow
    pre!!.next = null
    val l1 = sortList(head)
    val l2 = sortList(head2)
    return mergeTwoLists(l1, l2)
}

// 160. 相交链表
fun getIntersectionNode(headA: ListNode?, headB: ListNode?): ListNode? {
    val fakeHeadA = ListNode(-1)
    fakeHeadA.next = headA
    var lenA = 0
    var nodeA = fakeHeadA

    val fakeHeadB = ListNode(-1)
    fakeHeadB.next = headB
    var lenB = 0
    var nodeB = fakeHeadB

    while (nodeA.next != null || nodeB.next != null) {
        nodeA.next?.let {
            lenA++
            nodeA = it
        }
        nodeB.next?.let {
            lenB++
            nodeB = it
        }
    }
    if (nodeA != nodeB) return null

    nodeA = fakeHeadA
    nodeB = fakeHeadB
    while (lenA != lenB) {
        if (lenA > lenB) {
            lenA--
            nodeA = nodeA.next!!
        } else {
            lenB--
            nodeB = nodeB.next!!
        }
    }
    while (lenA != 0) {
        nodeA = nodeA.next!!
        nodeB = nodeB.next!!
        lenA--
        if (nodeA == nodeB) return nodeA
    }
    return null
}

// 5. 最长回文子串
fun longestPalindrome(s: String): String {
    var ans = ""
    val charArray = CharArray(s.length * 2 - 1) { '#' }
    for (idx in s.indices) {
        charArray[idx * 2] = s[idx]
    }
    for (centerIdx in charArray.indices) {
        val maxR = Math.min(centerIdx, charArray.size - centerIdx - 1)
        for (r in 0..maxR) {
            if (charArray[centerIdx - r] == charArray[centerIdx + r]) {
                if ((if (charArray[centerIdx] == '#') (r + 1) / 2 * 2 else r / 2 * 2 + 1) > ans.length) {
                    ans = charArray.sliceArray(centerIdx - r..centerIdx + r).filter { it != '#' }.joinToString("")
                }
            } else break
        }
    }
    return ans
}

// 283. 移动零
fun moveZeroes(nums: IntArray): Unit {
    var zeroIdx = 0
    while (zeroIdx < nums.size && nums[zeroIdx] != 0) {
        zeroIdx++
    }
    var rightIdx = zeroIdx + 1
    while (rightIdx < nums.size) {
        if (nums[rightIdx] != 0) {
            findKthLargest_swap(nums, zeroIdx, rightIdx)
            zeroIdx++
        }
        rightIdx++
    }
}

// 16. 最接近的三数之和
fun threeSumClosest(nums: IntArray, target: Int): Int {
    nums.sort()
    var ans = nums[0] + nums[1] + nums[2]
    for (i in nums.indices) {
        if (i != 0 && nums[i] == nums[i - 1]) continue
        val num1 = nums[i]
        var left = i + 1
        var right = nums.size - 1
        while (left < right) {
            val sum = num1 + nums[left] + nums[right]
            if (sum == target) return target
            else if (sum > target) {
                right--
            } else {
                left++
            }
            if (Math.abs(target - ans) > Math.abs(sum - target)) ans = sum
        }
    }
    return ans
}

// 322. 零钱兑换
fun coinChange(coins: IntArray, amount: Int): Int {
    val dp = IntArray(amount + 1) { Int.MAX_VALUE }
    dp[0] = 0
    for (i in dp.indices) {
        for (coin in coins) {
            val preAmount = i - coin
            if (preAmount >= 0 && dp[preAmount] != Int.MAX_VALUE) {
                dp[i] = Math.min(dp[i], dp[preAmount] + 1)
            }
        }
    }
    return if (dp[amount] == Int.MAX_VALUE) -1 else dp[amount]
}

// 518. 零钱兑换 II
fun change(amount: Int, coins: IntArray): Int {
    val dp = IntArray(amount + 1) { 0 }
    dp[0] = 1
    for (coin in coins) {
        for (i in coin until dp.size) {
            dp[i] = dp[i] + dp[i - coin]
        }
    }
    return dp[amount]
}