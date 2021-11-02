
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
fun main () {
    val array = intArrayOf(1,2,3)
    val intArray1 = intArrayOf(9,1)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1,2,3,4,5), intArrayOf(6,7,8,9,10), intArrayOf(11,12,13,14,15), intArrayOf(16,17,18,19,20), intArrayOf(21,22,23,24,25))
    val arrayOfStrings = arrayOf("Hello","Alaska","Dad","Peace")
    val ans = findWords(arrayOfStrings)
    println(ans.contentToString())
}