import java.util.*
import kotlin.math.sqrt

// 492. 构造矩形
fun constructRectangle(area: Int): IntArray {
    var width = sqrt(area.toDouble()).toInt()
    while (width > 0) {
        if (area % width == 0) return intArrayOf(area / width, width)
        width--
    }
    return intArrayOf(area, 1)
}

// 240. 搜索二维矩阵 II
fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
    var x = 0
    var y = matrix[0].size - 1
    while (x >= 0 && y < matrix.size) {
        val thisVal = matrix[x][y]
        if (thisVal == target) return true
        else if (thisVal > target) {
            y--
        }
        else {
            x++
        }
    }
    return false
}

// 496. 下一个更大元素 I
fun nextGreaterElement(nums1: IntArray, nums2: IntArray): IntArray {
    val hashMap = hashMapOf<Int, Int>()
    val decreaseStack = Stack<Int>()
    for (num in nums2) {
        while (decreaseStack.isNotEmpty()) {
            val topNum = decreaseStack.peek()
            if (topNum >= num) {
                break
            }
            else {
                hashMap[topNum] = num
                decreaseStack.pop()
            }
        }
        decreaseStack.push(num)
    }
    val ans = mutableListOf<Int>()
    for (num in nums1) {
        ans.add(hashMap[num] ?: -1)
    }
    return ans.toIntArray()
}

// 231. 2 的幂
fun isPowerOfTwo(n: Int): Boolean {
    return n > 0 && n.and(n-1) == 0
}
fun main () {
    val array = intArrayOf(1,2,3)
    val intArray1 = intArrayOf(9,1)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1,2,3,4,5), intArrayOf(6,7,8,9,10), intArrayOf(11,12,13,14,15), intArrayOf(16,17,18,19,20), intArrayOf(21,22,23,24,25))
    val arrayOfStrings = arrayOf("Science","is","what","we","understand","well","enough","to","explain",
        "to","a","computer.","Art","is","everything","else","we","do")
    val ans = searchMatrix(arrayIntArray, 20)
    println(ans)
}