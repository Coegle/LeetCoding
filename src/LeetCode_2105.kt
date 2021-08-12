import java.util.*
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.math.max

// 1720. 解码异或后的数组
fun decode(encoded: IntArray, first: Int): IntArray {
    val ret = IntArray(encoded.size + 1)
    ret[0] = first
    for (i in encoded.indices) {
        ret[i + 1] = encoded[i] xor ret[i]
    }
    return ret
}
// 1482. 制作 m 束花所需的最少天数
fun mindaysCheck(bloomDay: IntArray, minDay: Int, m: Int, k: Int): Boolean {
    var cnt = 0 // 连续朵数
    var totalFlowers = 0 // 满足的总数
    for(day in bloomDay) {
        if (day > minDay) { // 不满足条件
            cnt = 0
        }
        else {
            cnt++
        }
        if (cnt == k) { // 生成下一束
            totalFlowers++
            if (totalFlowers >= m) return true
            cnt = 0
        }
    }
    return false
}
fun minDays(bloomDay: IntArray, m: Int, k: Int): Int {
    var rightDay = bloomDay.maxByOrNull { it }!!
    var leftDay = bloomDay.minByOrNull { it }!!
    var ans = -1
    if (m*k > bloomDay.size) return -1
    while (leftDay <= rightDay) {
        val mid = (leftDay + rightDay) / 2
        if (mindaysCheck(bloomDay, mid, m, k)) {
            ans = mid
            rightDay = mid - 1
        }
        else {
            leftDay = mid + 1
        }
    }
    return ans
}

// 872. 叶子相似的树
class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}
fun leafSimilarDfs(root: TreeNode?, serial: MutableList<Int>) {
    if (root == null) return
    val serial = serial
    if (root.left == null && root.right == null) {
        serial.add(root.`val`)
    }
    if (root.left != null) leafSimilarDfs(root.left, serial)
    if (root.right != null) leafSimilarDfs(root.right, serial)
}

fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
    val serial1 = mutableListOf<Int>()
    val serial2 = mutableListOf<Int>()
    leafSimilarDfs(root1, serial1)
    leafSimilarDfs(root2, serial2)
    return serial1 == serial2
}
// 1734. 解码异或后的排列
fun decode(encoded: IntArray): IntArray {
    var fullXor = encoded.size + 1
    var xorWithoutFirst = 0
    for (i in encoded.indices) {
        fullXor = fullXor xor (i + 1)
        if (i % 2 == 1) {
            xorWithoutFirst = xorWithoutFirst xor encoded[i]
        }
    }
    val first = fullXor xor xorWithoutFirst
    return decode(encoded, first)
}
// 1310. 子数组异或查询
fun xorQueries(arr: IntArray, queries: Array<IntArray>): IntArray {
    val xorArray = IntArray(arr.size)
    xorArray[0] = arr[0]
    for(i in 1 until arr.size) {
        xorArray[i] = xorArray[i-1] xor arr[i]
    }
    val ans = IntArray(queries.size)
    for (i in queries.indices) {
        val left = queries[i][0]
        val right = queries[i][1]
        ans[i] = xorArray[right]
        if (left != 0) {
            ans[i] = ans[i] xor xorArray[left - 1]
        }
    }
    return ans
}
// 1269. 停在原地的方案数
fun numWays(steps: Int, arrLen: Int): Int {
    if (arrLen == 1 || steps == 1) return 1
    val mod = 1000000007
    val minJ = Math.min(steps, arrLen)
    val array = Array(steps+1) {IntArray(minJ)}
    array[0][0] = 1
    for(i in 1 until array.size) {
        array[i][0] = (array[i-1][0] + array[i-1][1]) % mod
        for(j in 1 until minJ - 1) {
            array[i][j] = ((array[i-1][j] + array[i-1][j-1]) % mod + array[i-1][j+1]) % mod
        }
        array[i][minJ-1] = (array[i-1][minJ-1] + array[i-1][minJ-2]) % mod
    }
    return array[steps][0]
}
// 12. 整数转罗马数字
fun intToRoman(num: Int): String {
    val map = mapOf(1000 to "M", 900 to "CM", 500 to "D", 400 to "CD", 100 to "C", 90 to "XC", 50 to "L", 40 to "XL", 10 to "X", 9 to "IX", 5 to "V", 4 to "IV", 1 to "I")
    var remain = num

    var ans = ""
    for ((num, romanNum) in map) {
        val repeatNum = remain / num
        if (repeatNum != 0) {
            remain %= num
            repeat(repeatNum) {
                ans+= romanNum
            }
        }
    }
    return ans
}
// 229. 求众数 II
fun majorityElement(nums: IntArray): List<Int> {
    var ans1 = nums[0]
    var cnt1 = 0
    var ans2 = nums[0]
    var cnt2 = 0
    for (num in nums) {
        when {
            num == ans1 -> {
                cnt1++
            }
            num == ans2 -> {
                cnt2++
            }
            cnt1 == 0 -> {
                ans1 = num
                cnt1 = 1
            }
            cnt2 == 0 -> {
                ans2 = num
                cnt2 = 1
            }
            else -> {
                cnt1--
                cnt2--
            }
        }
    }
    var vali1 = 0
    var vali2 = 0
    for (num in nums) {
        if (num == ans1) vali1++
        else if (num == ans2) vali2++ // 如果整个数列相等的情况下 ans1 == ans2
    }
    val ansList = mutableListOf<Int>()
    if (vali1 > nums.size / 3) ansList.add(ans1)
    if (vali2 > nums.size / 3) ansList.add(ans2)
    return ansList
}
// 13. 罗马数字转整数
fun romanToInt(s: String): Int {
    val map = mapOf("M" to 1000, "CM" to 900, "D" to 500, "CD" to 400, "C" to 100, "XC" to 90, "L" to 50, "XL" to 40, "X" to 10, "IX" to 9, "V" to 5, "IV" to 4, "I" to 1)
    val list = listOf('C', 'X', 'I')
    var ans = 0
    var i = 0
    while (i < s.length) {
        ans += if (list.contains(s[i]) && i + 1 < s.length && map.containsKey(s.substring(i..i+1))) {
            map[s.substring(i..++i)]!!
        } else {
            map[s[i].toString()]!!
        }
        i++
    }
    return ans
}
// 421. 数组中两个数的最大异或值
fun findMaximumXOR(nums: IntArray): Int {
    if (nums.size == 1) return 0

    var ans = 0
    for (i in 30 downTo 0) {
        val tempAns = ans or 1
        val hashMap = HashMap<Int, Int>()
        for (num in nums) {
            hashMap[num shr i] = 1
        }
        for (num in nums) {
             val checkJ = tempAns xor (num shr i)
            if (hashMap.containsKey(checkJ)) {
                ans = (ans or 1)
                break
            }
        }
        if (i != 0) ans = ans shl 1
    }
    return ans
}
fun main () {
    val array = intArrayOf(3)
    val intArray1 = intArrayOf(2,3,7,6)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1), intArrayOf(0,2,4), intArrayOf(1,3,4), intArrayOf(2), intArrayOf(1,2))
    val ans = findMaximumXOR(intArray1)
    println(ans)
}