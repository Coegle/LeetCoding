import kotlin.math.max

// 395. 至少有 K 个重复字符的最长子串
fun longestSubstring(s: String, k: Int): Int {
    if (s.length < k) return 0
    val charCnt = mutableMapOf<Char, Int>()
    s.forEach { c ->
        charCnt[c] = charCnt.getOrDefault(c, 0) + 1
    }
    val chars = mutableSetOf<Char>()
    charCnt.forEach { (key, value) ->
        if (value < k) chars.add(key)
    }
    if (chars.size == 0) return s.length
    val subStrings = s.split(*chars.toCharArray())
    var ans = 0
    subStrings.forEach {
        var left = 0
        var right = it.length - 1
        while (left <= right) {
            if (chars.contains(it[left])) {
                left++
            } else break
            if (chars.contains(it[right])) {
                right--
            } else break
        }
        if (left <= right) ans = max(ans, longestSubstring(it.substring(left..right), k))
    }
    return ans
}

// 1763. 最长的美好子字符串
fun longestNiceSubstring(s: String): String {
    if (s.length <= 1) return ""
    val lowerCaseChars = mutableSetOf<Char>()
    val upperCaseChars = mutableSetOf<Char>()
    for (c in s) {
        if (c in 'a'..'z') lowerCaseChars.add(c)
        else if (c in 'A'..'Z') upperCaseChars.add(c.lowercaseChar())
    }
    if (lowerCaseChars == upperCaseChars) return s
    val chars =
        (lowerCaseChars subtract upperCaseChars) union (upperCaseChars subtract lowerCaseChars).map { it.uppercaseChar() }
    val subStrings = s.split(*chars.toCharArray())
    var ans = ""
    subStrings.forEach {
        val resStr = longestNiceSubstring(it)
        ans = if (resStr.length > ans.length) resStr else ans
    }
    return ans
}

// 2000. 反转单词前缀
fun reversePrefix(word: String, ch: Char): String {
    val stack = ArrayDeque<Char>()
    var idx = 0
    while (idx < word.length) {
        stack.addFirst(word[idx])
        if (word[idx] == ch) break
        idx++
    }
    if (idx == word.length) return word
    return stack.joinToString(separator = "", transform = { it.toString() }) + word.substring(idx + 1)
}

// 1414. 和为 K 的最少斐波那契数字数目
fun findMinFibonacciNumbers(k: Int): Int {
    val fibos = ArrayDeque(listOf(2, 1))
    while (fibos.first() < k) {
        val nxt = fibos[0] + fibos[1]
        if (nxt > k) break
        fibos.addFirst(nxt)
    }
    var ans = 0
    var remainK = k
    for (num in fibos) {
        while (num <= remainK) {
            remainK -= num
            ans++
        }
        if (remainK == 0) break
    }
    return ans
}