import java.lang.Integer.min
import java.util.*
import kotlin.collections.HashSet
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

// 869. 重新排序得到 2 的幂
fun reorderedPowerOf2(n: Int): Boolean {
    val numbers = Array(31) { 1.shl(it) }.toHashSet()
    val cntHash = HashSet<Int>()
    for (num in numbers) {
        val hash = reorderedPowerOf2ToGetHash(num)
        cntHash.add(hash)
    }
    val nHash = reorderedPowerOf2ToGetHash(n)
    return cntHash.contains(nHash)
}
fun reorderedPowerOf2ToGetHash(n: Int): Int {
    var number = n
    val cntArray = Array(10) {0}
    while (number > 0) {
        cntArray[number % 10]++
        number = number.div(10)
    }
    var hash = 0
    for (cnt in cntArray) {
        hash = (hash + cnt) * 10
    }
    return hash
}

// 335. 路径交叉
fun isSelfCrossing(distance: IntArray): Boolean {
    if (distance.size < 4) return false
    for (i in 3 until distance.size) {
        if (distance[i] >= distance[i-2] && distance[i-1] <= distance[i-3]) return true
        if (i >= 4 && distance[i-1] == distance[i-3] && distance[i] + distance[i-4] >= distance[i-2]) return true
        if (i >= 5 && distance[i-1] + distance[i-5] >= distance[i-3] && distance[i-1] <= distance[i-3] && distance[i-2] > distance[i-4] && distance[i-2] <= distance[i] + distance[i-4]) return true
    }
    return false
}

// 260. 只出现一次的数字 III
fun singleNumber(nums: IntArray): IntArray {
    var xorResult = 0
    nums.forEach { xorResult = xorResult.xor(it) }
    var k = 0
    for (i in 0 until 32) {
        if (xorResult.and(1.shl(i)) != 0) {
            k = i
            break
        }
    }
    val ans = IntArray(2)
    nums.forEach {
        if (it.and(1.shl(k)) != 0) {
            ans[0] = ans[0].xor(it)
        }
        else ans[1] = ans[1].xor(it)
    }
    return ans
}

// 500. 键盘行
fun findWords(words: Array<String>): Array<String> {
    val charAtRow = "12210111011122000010020202"
    val ans = mutableListOf<String>()
    for (word in words) {
        var flag = 0
        val row = charAtRow[word[0].lowercaseChar() - 'a']
        for (c in word) {
            if (charAtRow[c.lowercaseChar() - 'a'] != row) {
                flag = 1
                break
            }
        }
        if (flag == 0) {
            ans.add(word)
        }
    }
    return ans.toTypedArray()
}

// 575. 分糖果
fun distributeCandies(candyType: IntArray): Int {
    val set = candyType.distinct()
    return min(candyType.size / 2, set.size)
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