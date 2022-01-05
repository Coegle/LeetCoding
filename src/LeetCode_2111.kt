import java.util.*
import kotlin.math.min

// 237. 删除链表中的节点
fun deleteNode(node: ListNode?) {
    var thisNode = node
    while (thisNode != null) {
        val nextNode = thisNode.next
        if (nextNode != null) {
            thisNode.`val` = nextNode.`val`
            if (nextNode.next == null) {
                thisNode.next = null
                break
            }
        }
        thisNode = nextNode
    }
}

// 1576. 替换所有的问号
fun modifyString(s: String): String {
    val ans = s.toCharArray()
    val nxtChars = charArrayOf('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p' ,'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    var nxtIdx = 0
    for (i in ans.indices) {
        if (ans[i] == '?') {
            while (i - 1 >= 0 && ans[i - 1] == nxtChars[nxtIdx] || i + 1 < ans.size && ans[i + 1] == nxtChars[nxtIdx]) {
                nxtIdx = (nxtIdx + 1) % 26
            }
            ans[i] = nxtChars[nxtIdx]
        }
    }
    return String(ans)
}
fun main () {
    val array = intArrayOf(1,2,3)
    val intArray1 = intArrayOf(0,1,0,2,1,0,1,3,2,1,2,1)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1,2,3,4,5), intArrayOf(6,7,8,9,10), intArrayOf(11,12,13,14,15), intArrayOf(16,17,18,19,20), intArrayOf(21,22,23,24,25))
    val arrayOfStrings = arrayOf("Hello","Alaska","Dad","Peace")
    val ans = modifyString("j?qg??b")
    println(ans)
}