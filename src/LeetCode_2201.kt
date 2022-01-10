import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashSet

// 71. 简化路径
fun simplifyPath(path: String): String {
    val dirs = path.split('/')
    val stack: Deque<String> = LinkedList()
    for (dir in dirs) {
        if (dir == "..") {
            if (stack.isNotEmpty()) stack.removeLast()
        }
        else if (dir == "." || dir == "") {
            continue
        }
        else {
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
fun grayCodeDfs(ans: MutableList<Int>, n: Int, vis: HashSet<Int>): Boolean{
    if (1.shl(n) == ans.size) return true
    for (i in 0 until n) {
        val next = ans.last().xor(1.shl(i))
        if (!vis.contains(next)) {
            ans.add(next)
            vis.add(next)
            if(grayCodeDfs(ans, n, vis)) return true
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
            while(begin + nxt.length <= num.length) {
                if (num.slice(begin until begin + nxt.length).toLong() == nxt.toLong()) {
                    begin += nxt.length
                    val tmp = secondNum.toLong()
                    secondNum = nxt
                    nxt = (tmp + nxt.toLong()).toString()
                }
                else {
                    break
                }
            }
            if (begin == num.length) return true
        }
    }
    return false
}

fun main() {
    val num = 3
    val ans = isAdditiveNumber("198019823962")
    println(ans)
}