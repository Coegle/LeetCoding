import java.util.*

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
fun main () {
    val array = intArrayOf(3)
    val intArray1 = intArrayOf(-1,0,3,5,9,12)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1), intArrayOf(0,2,4), intArrayOf(1,3,4), intArrayOf(2), intArrayOf(1,2))
    val ans = search(intArray1, 2)
    println(ans)
}