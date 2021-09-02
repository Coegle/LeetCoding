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