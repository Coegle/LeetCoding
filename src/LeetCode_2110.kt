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

fun main () {
    val array = intArrayOf(1,2,3)
    val intArray1 = intArrayOf(9,1)
    val intList1 = mutableListOf(1, 2, 3)
    val intList2 = mutableListOf(1, 2, 3)
    val str = "cbbd"
    val arrayIntArray = arrayOf(intArrayOf(1), intArrayOf(0,2,4), intArrayOf(1,3,4), intArrayOf(2), intArrayOf(1,2))
    val arrayOfStrings = arrayOf("Science","is","what","we","understand","well","enough","to","explain",
        "to","a","computer.","Art","is","everything","else","we","do")
    val ans = constructRectangle(17)
    println(ans.contentToString())
}