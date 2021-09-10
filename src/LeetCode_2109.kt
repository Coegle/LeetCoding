import java.util.*
import kotlin.Comparator

class ListNode(var `val`: Int) {
    var next: ListNode? = null
}
// 剑指 Offer 22. 链表中倒数第k个节点
fun getKthFromEnd(head: ListNode?, k: Int): ListNode? {
    val stack = Stack<ListNode>()
    var node = head
    while (node != null) {
        stack.push(node)
        node = node.next
    }
    repeat(k-1) {
        stack.pop()
    }
    return stack.peek()
}
// 面试题 17.14. 最小K个数
fun smallestK(arr: IntArray, k: Int): IntArray {
    val queue = PriorityQueue<Int>()
    for (i in arr) {
        queue.add(i)
    }
    var k1 = k
    val ans = mutableListOf<Int>()
    while (queue.isNotEmpty() && k1 != 0) {
        ans.add(queue.remove())
        k1--
    }
    return ans.toIntArray()
}

// 704. 二分查找
fun search(nums: IntArray, target: Int): Int {
    var left = 0
    var right = nums.size - 1
    while (left <= right) {
        val mid = left + (right - left) / 2
        when {
            nums[mid] == target -> {
                return mid
            }
            nums[mid] > target -> {
                right = mid - 1
            }
            else -> {
                left = mid + 1
            }
        }
    }
    return -1
}

// 1221. 分割平衡字符串
fun balancedStringSplit(s: String): Int {
    var ans = 0
    var bias = 0
    for (c in s) {
        if (c == 'L') {
            bias++
        }
        else {
            bias--
        }
        if (bias == 0) {
            ans++
        }
    }
    return ans
}

// 502. IPO
fun findMaximizedCapital(k: Int, w: Int, profits: IntArray, capital: IntArray): Int {
    data class Project(val capital: Int, val profit: Int)
    val comparator1: Comparator<Project> = compareBy { it.capital }
    val comparator2: Comparator<Project> = compareBy { -it.profit }
    val notQualifiedQueue = PriorityQueue(comparator1)
    val maxProfitQueue = PriorityQueue(comparator2)
    for (idx in profits.indices) {
        notQualifiedQueue.add(Project(capital[idx], profits[idx]))
    }
    var maxCapital = w
    var projectNum = k
    while (projectNum != 0) {
        // 挑出当次符合资格的项目
        while (notQualifiedQueue.isNotEmpty() && notQualifiedQueue.peek().capital <= maxCapital) {
            maxProfitQueue.add(notQualifiedQueue.remove())
        }
        // 挑出当次最有价值的项目
        if (maxProfitQueue.isNotEmpty()) {
            maxCapital += maxProfitQueue.remove().profit
            projectNum--
        }
        else {
            break
        }
    }
    return maxCapital
}
// 68. 文本左右对齐
fun fullJustify(words: Array<String>, maxWidth: Int): List<String> {
    val paragraphs: MutableList<MutableList<String>> = mutableListOf()
    // 分行
    var width = 0
    var paragraph: MutableList<String> = mutableListOf()
    var idx = 0
    while (idx < words.size) {
        while (idx < words.size && words[idx].length + width <= maxWidth) {
            paragraph.add(words[idx])
            width += words[idx].length + 1
            idx++
        }
        width = 0
        paragraphs.add(paragraph)
        paragraph = mutableListOf()
    }
    // 处理每行（除末行）
    idx = 0
    val ans = mutableListOf<String>()
    while (idx < paragraphs.size - 1) {
        val thisParagraph = paragraphs[idx]
        var string = ""
        if (thisParagraph.size == 1) {
            string = thisParagraph[0] + " ".repeat(maxWidth - thisParagraph[0].length)
        }
        else {
            var totalWidth = maxWidth
            thisParagraph.forEach { totalWidth -= it.length }
            val commonWidth = totalWidth / (thisParagraph.size - 1)

            var leftWidth = totalWidth % (thisParagraph.size - 1)
            println("$commonWidth, $leftWidth")
            var idx2 = 0
            while (idx2 < thisParagraph.size - 1) {
                string += thisParagraph[idx2] + " ".repeat(commonWidth + if (leftWidth == 0) 0 else 1)
                leftWidth = if (leftWidth != 0) leftWidth - 1 else 0
                idx2++
            }
            string += thisParagraph[idx2]
        }
        ans.add(string)
        idx++
    }
    // 处理末尾段落
    val lastParagraph = paragraphs[idx]
    var lastStr = ""
    for (i in 0..lastParagraph.size - 2) {
        lastStr += lastParagraph[i] + " "
    }
    lastStr += lastParagraph.last()
    lastStr += " ".repeat(maxWidth - lastStr.length)
    ans += lastStr
    return ans
}

// 1894. 找到需要补充粉笔的学生编号
fun chalkReplacer(chalk: IntArray, k: Int): Int {
    var cycle = 0
    for (i in chalk.indices) {
        cycle += chalk[i]
        // 如果累加过程中大于粉笔数 K，则说明在第一轮的当前已经用完，直接返回当前 index，无需继续累加，可以防止数据过大导致溢出
        if (cycle > k) {
            return i
        }
    }
    var remain = k % cycle
    var ans = 0
    while (remain >= 0) {
        remain -= chalk[ans]
        ans++
    }
    return ans - 1
}
fun main () {
    val array = intArrayOf(1,2,3)
    val intArray1 = intArrayOf(0,1,2)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1), intArrayOf(0,2,4), intArrayOf(1,3,4), intArrayOf(2), intArrayOf(1,2))
    val arrayOfStrings = arrayOf("Science","is","what","we","understand","well","enough","to","explain",
                "to","a","computer.","Art","is","everything","else","we","do")
    val ans = fullJustify(arrayOfStrings, 20)
    println(ans)
}