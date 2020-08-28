## Array

### 11. Contain the most rain water

![img](/Users/jyl/Documents/notes/question_11.png)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        max_area = 0
        while left < right:
            area = min(height[left], height[right]) * (right - left)
            max_area = max(area, max_area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area
```



### 42. Trapping Rain Water

Given *n* non-negative integers representing an elevation map where the width of  each bar is 1, compute how much water it is able to trap after raining.

![img](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)
 The above elevation map is represented by array  [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue  section) are being trapped. **Thanks Marcos** for contributing this image!

**Example:**

```
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

```python	
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height: return 0
        
        left, right = 0, len(height)-1
        max_left = max_right = 0
        res = 0
        while left < right:
            if height[left] <= height[right]:
                max_left = max(max_left, height[left])
                res += (max_left - height[left])
                left += 1
            elif height[left] > height[right]:
                max_right = max(max_right, height[right])
                res += (max_right - height[right])
                right -= 1
                
        return res
```



### 15. 3 Sum

Given an array `nums` of *n* integers, are there elements *a*, *b*, *c* in `nums` such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:**

The solution set must not contain duplicate triplets.

**Example:**

```
Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        res = set()
        nums.sort()
        for i, v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i-1]:
                continue
            d = {}
            for x in nums[i+1:]:
                if x not in d:
                    d[-v-x] = 1
                else:
                    res.add((v, x, -v-x))
        return list(res)
```



### 26. Remove Duplicates from Sorted Array

Given a sorted array *nums*, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that each element appear only *once* and return the new length.

Do not allocate extra space for another array, you must do this by **modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** with O(1) extra memory.

**Example 1:**

```
Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
```

**Example 2:**

```
Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length.
```



```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums: return
        store = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                store += 1
                nums[store] = nums[i]
        return store + 1
```



### 66. Plus One

Given a **non-empty** array of digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the  head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

 

**Example 1:**

```
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
```

**Example 2:**

```
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
```

**Example 3:**

```
Input: digits = [0]
Output: [1]
```

 

**Constraints:**

- `1 <= digits.length <= 100`
- `0 <= digits[i] <= 9`

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        
        if not digits:
            return
        
        if len(digits) == 1 and digits[-1] == 9:
            return [1, 0]
        
        if digits[-1] < 9:
            digits[-1] += 1
            return digits
            
        if digits[-1] == 9:
            return self.plusOne(digits[:-1]) + [0]
```



### 88. Merge Sorted Array

Given two sorted integer arrays *nums1* and *nums2*, merge *nums2* into *nums1* as one sorted array.

**Note:**

- The number of elements initialized in *nums1* and *nums2* are *m* and *n* respectively.
- You may assume that *nums1* has enough space (size that is **equal** to *m* + *n*) to hold additional elements from *nums2*.

**Example:**

```
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]
```

 

**Constraints:**

- `-10^9 <= nums1[i], nums2[i] <= 10^9`
- `nums1.length == m + n`
- `nums2.length == n`

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums2[n-1] > nums1[m-1]:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
            else:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
```



### 169. Majority Element

Given an array of size *n*, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.

You may assume that the array is non-empty and the majority element always exist in the array.

**Example 1:**

```
Input: [3,2,3]
Output: 3
```

**Example 2:**

```
Input: [2,2,1,1,1,2,2]
Output: 2
```

```python	
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count, major = 0, None
        for num in nums:
            if count > len(nums) // 2:
                return major
            if count == 0:
                major = num
                count += 1
            elif num == major:
                count += 1
            else:
                count -= 1
        return major
        
```



### 189. Rotate Array

Given an array, rotate the array to the right by *k* steps, where *k* is non-negative.

**Follow up:**

- Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
- Could you do it in-place with O(1) extra space?

 

**Example 1:**

```
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
```

**Example 2:**

```
Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]
```

 

**Constraints:**

- `1 <= nums.length <= 2 * 10^4`
- It's guaranteed that `nums[i]` fits in a 32 bit-signed integer.
- `k >= 0`

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums or k % len(nums) == 0:
            return
        
        def reverse(l: int, r:int) -> None:
            if l >= r:
                return
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
                
        k = k % len(nums)
        reverse(0, len(nums)-1)
        reverse(0, k-1)
        reverse(k, len(nums)-1)
```



### 238. Product of Array Except Self

Given an array `nums` of *n* integers where *n* > 1,  return an array `output` such that `output[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

**Example:**

```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

**Constraint:** It's guaranteed that the product of the  elements of any prefix or suffix of the array (including the whole  array) fits in a 32 bit integer.

**Note:** Please solve it **without division** and in O(*n*).

**Follow up:**
 Could you solve it with constant space complexity? (The output array **does not** count as extra space for the purpose of space complexity analysis.)

```python
class Solution(object):
    def productExceptSelf(self, nums):
        result = [1 for i in nums]
        nf = 1
        nb = 1
        length = len(nums)
        for i in range(length):
            result[i] *= nf
            nf *= nums[i]
            result[length-i-1] *= nb
            nb *= nums[length-i-1]
        return result
```



### 283. Move Zeroes

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Example:**

```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```

**Note**:

1. You must do this **in-place** without making a copy of the array.
2. Minimize the total number of operations.

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums: return
        store = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[store] = nums[store], nums[i]
                store += 1
```



### 560. Subarray Sum Equals K

Given an array of integers and an integer **k**, you need to find the total number of continuous subarrays whose sum equals to **k**.

**Example 1:**

```
Input:nums = [1,1,1], k = 2
Output: 2
```

 

**Constraints:**

- The length of the array is in range [1, 20,000].
- The range of numbers in the array is [-1000, 1000] and the range of the integer **k** is [-1e7, 1e7].

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        sum_count = collections.defaultdict(int)
        sum_count[0] = 1  # 这里一定要等于1
        sum = res = 0
        
        for num in nums:
            sum += num
            if sum - k in sum_count:
                res += sum_count[sum - k]
            sum_count[sum] += 1
        return res
```



### 509. Fibonacci Number

The **Fibonacci numbers**, commonly denoted `F(n)` form a sequence, called the **Fibonacci sequence**, such that each number is the sum of the two preceding ones, starting from `0` and `1`. That is,

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), for N > 1.
```

Given `N`, calculate `F(N)`.

 

**Example 1:**

```
Input: 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.
```

**Example 2:**

```
Input: 3
Output: 2
Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.
```

**Example 3:**

```
Input: 4
Output: 3
Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.
```

 

**Note:**

0 ≤ `N` ≤ 30.

```python	
# recursion, dp
class Solution:
    def fib(self, N: int) -> int:
        if N < 2:
            return N
        pre, cur = 0, 1
        for i in range(1, N):    # from 1 to N-1时的pre，cur时N-1+1，也就是N
            pre, cur = cur, pre + cur
        return cur
```



## Backtrack

\17. Letter Combinations of a Phone Number

Medium

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Telephone-keypad2.svg/200px-Telephone-keypad2.svg.png)

**Example:**

```
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

**Note:**

Although the above answer is in lexicographical order, your answer could be in any order you want.

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        dicty = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']
                }

        def backtrack(index, track):
            # if index == len(digits):
            if len(track) == len(digits):
                res.append(track)
                return
                
            for item in dicty[digits[index]]:
                backtrack(index + 1, track + item)
            
        res = []
        backtrack(0, '')
        return res
```



### 22. Generate Parentheses

Medium

Given *n* pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given *n* = 3, a solution set is:

```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        self.list = []
        self._gen(0, 0, n, "")
        return self.list
    
    def _gen(self, left, right, n, result):
        if left == n and right == n:
            self.list.append(result)
            print(result)
            return
        
        if left < n:
            self._gen(left + 1, right, n, result + '(')
        if right < left and right < n:
            self._gen(left, right + 1, n, result + ')')
```



### 31. Next Permutation

Medium

Implement **next permutation**, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be **[in-place](http://en.wikipedia.org/wiki/In-place_algorithm)** and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

```
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
```

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        left, right = 0, len(nums)-1
        for i in range(len(nums)-1, 0, -1):
            if nums[i] > nums[i-1]:
                left = i
                for j in range(len(nums)-1, i-1, -1):
                    if nums[j] > nums[i-1]:
                        nums[i-1], nums[j] = nums[j], nums[i-1]
                        break
                break
                
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```



### 39. Combination Sum

Medium

Given a **set** of candidate numbers (`candidates`) **(without duplicates)** and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The **same** repeated number may be chosen from `candidates` unlimited number of times.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```

**Example 2:**

```
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

**Constraints:**

- `1 <= candidates.length <= 30`
- `1 <= candidates[i] <= 200`
- Each element of `candidate` is unique.
- `1 <= target <= 500`

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(start, path):
            if sum([p for p in path]) >= target:
                if sum([p for p in path]) == target:
                    res.append(path)
                return
                
            for i in range(start, len(candidates)):
                backtrack(i, path+[candidates[i]])

        backtrack(0, [])
        return res
```



### 40. Combination Sum II

Medium

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

**Example 2:**

```
Input: candidates = [2,5,2,1,2], target = 5,
A solution set is:
[
  [1,2,2],
  [5]
]
```

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        res = []
        candidates.sort()
        
        def backtrack(start, path):
            if sum([p for p in path]) >= target:
                if sum([p for p in path]) == target:
                    res.append(path)
                return
            
            seen = set()
            for i in range(start, len(candidates)):
                if candidates[i] in seen:
                    continue
                seen.add(candidates[i])
                backtrack(i+1, path+[candidates[i]])
            
        backtrack(0, [])
        return res        
```



### 216. Combination Sum III

Medium

Find all possible combinations of ***k*** numbers that add up to a number ***n***, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

**Note:**

- All numbers will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: k = 3, n = 7
Output: [[1,2,4]]
```

**Example 2:**

```
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
```

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        if k <= 0 or n <= 0:
            return []
        if n > sum([i for i in range(9, 9-k, -1)]):
            return []

        def backtrack(start, track):
            if len(track) == k:
                if sum([i for i in track]) == n:
                    res.append(track.copy())
                return
            
            for i in range(start, 10):
                backtrack(i+1, track+[i])
         
        res = []
        backtrack(1, [])
        return res
```



### 46. Permutations

Medium

Given a collection of **distinct** integers, return all possible permutations.

**Example:**

```
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums, path):
            # if not nums:
            if len(path) == n:    
                res.append(path)
                return

            for i in range(0, len(nums)):
                backtrack(nums[:i]+nums[i+1:], path+[nums[i]])
                
        n = len(nums)     
        res = []        
        backtrack(nums, [])
        return res
        
```



### 47. Permutations II

Medium

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

**Example:**

```
Input: [1,1,2]
Output:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        if not nums: return []
        def backtrack(nums, track):
            if len(track) == n:
                res.append(track)
                return
            
            seen = set()
            for i in range(len(nums)):
                if nums[i] in seen:
                    continue
                seen.add(nums[i])
                backtrack(nums[:i] + nums[i+1:], track + [nums[i]])
                
        n = len(nums)
        res = []
        backtrack(nums, [])
        return res
```



### 51. N-Queens

Hard

The *n*-queens puzzle is the problem of placing *n* queens on an *n*×*n* chessboard such that no two queens attack each other.

![img](https://assets.leetcode.com/uploads/2018/10/12/8-queens.png)

Given an integer *n*, return all distinct solutions to the *n*-queens puzzle.

Each solution contains a distinct board configuration of the *n*-queens' placement, where `'Q'` and `'.'` both indicate a queen and an empty space respectively.

**Example:**

```
Input: 4
Output: [
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above.
```

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n < 1:
            return []
        self.result = []
        self.cols, self.pies, self.nas = set(), set(), set()
        self.DFS(n, 0, [])
        return self._generate_result(n)
                
    # current state: current result
    def DFS(self, n, row, current_state):
        if row >= n:
            self.result.append(current_state)
            return
        
        for col in range(n):
            if col in self.cols or col + row in self.pies or row - col in self.nas:
                continue
            
            self.cols.add(col)
            self.pies.add(row + col)
            self.nas.add(row - col)
            
            self.DFS(n, row + 1, current_state + [col])
            
            self.cols.remove(col)
            self.pies.remove(row + col)
            self.nas.remove(row - col)
            
    def _generate_result(self, n):
        board = []
        for res in self.result:
            for i in res:
                board.append('.' * i + 'Q' + '.' * (n - i - 1))
        return [board[i :i + n] for i in range(0, len(board), n)]
```



### 60. Permutation Sequence

Hard

The set `[1,2,3,...,*n*]` contains a total of *n*! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for *n* = 3:

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

Given *n* and *k*, return the *k*th permutation sequence.

**Note:**

- Given *n* will be between 1 and 9 inclusive.
- Given *k* will be between 1 and *n*! inclusive.

**Example 1:**

```
Input: n = 3, k = 3
Output: "213"
```

**Example 2:**

```
Input: n = 4, k = 9
Output: "2314"
```

```python
# https://leetcode-cn.com/problems/permutation-sequence/solution/golang-100-faster-by-a-bai-152/

import math
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        tokens = [str(i) for i in range(1, n+1)]
        res = ''
        k -= 1
        while n > 0:
            n -= 1
            a, k = divmod(k, math.factorial(n))
            res += tokens.pop(a)
        return res
```



### 77. Combinations

Medium

Given two integers *n* and *k*, return all possible combinations of *k* numbers out of 1 ... *n*.

You may return the answer in **any order**.

 

**Example 1:**

```
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

**Example 2:**

```
Input: n = 1, k = 1
Output: [[1]]
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def backtrack(start, path):
            if len(path) == k:
                res.append(path)           
            for i in range(start, n+1):
                backtrack(i+1, path + [i])
        backtrack(1, [])
        return res
```



### 78. Subsets

Medium

Given a set of **distinct** integers, *nums*, return all possible subsets (the power set).

**Note:** The solution set must not contain duplicate subsets.

**Example:**

```
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start, track):
            res.append(track)
            for i in range(start, len(nums)):
                backtrack(i+1, track+[nums[i]])
                
        res = []   
        backtrack(0, [])
        return res
```



### 90. Subsets II

Medium

Given a collection of integers that might contain duplicates, ***nums\***, return all possible subsets (the power set).

**Note:** The solution set must not contain duplicate subsets.

**Example:**

```
Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        
        def backtrack(start, path):
            res.append(path)
            seen = set()
            for i in range(start, len(nums)):
                if nums[i] in seen:
                    continue
                seen.add(nums[i])
                backtrack(i+1, path+[nums[i]])

        backtrack(0, [])
        return res
        
```



### 93. Restore IP Addresses

Medium

Given a string `s` containing only digits. Return all possible valid IP addresses that can be obtained from `s`. You can return them in **any** order.

A **valid IP address** consists of exactly four integers, each integer is between `0` and `255`, separated by single points and cannot have leading zeros. For example, "0.1.2.201" and "192.168.1.1" are **valid** IP addresses and "0.011.255.245", "192.168.1.312" and "192.168@1.1" are **invalid** IP addresses. 

 

**Example 1:**

```
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
```

**Example 2:**

```
Input: s = "0000"
Output: ["0.0.0.0"]
```

**Example 3:**

```
Input: s = "1111"
Output: ["1.1.1.1"]
```

**Example 4:**

```
Input: s = "010010"
Output: ["0.10.0.10","0.100.1.0"]
```

**Example 5:**

```
Input: s = "101023"
Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
```

 

**Constraints:**

- `0 <= s.length <= 3000`
- `s` consists of digits only.

```python
# track = [number1, number2, number3, number4]

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def backtrack(s, track):
            if len(s) == 0 and len(track) == 4:
                res.append('.'.join(track))
                return
            
            if len(track) < 4:
                for i in range(min(3, len(s))):
                    p, n = s[:i+1], s[i+1:]
                    if 0 <= int(p) <= 255 and str(int(p)) == p:
                        backtrack(n, track + [p])
   
        res = []            
        backtrack(s, [])
        return res
```



### 254. Factor Combinations

Medium

Numbers can be regarded as product of its factors. For example,

```
8 = 2 x 2 x 2;
  = 2 x 4.
```

Write a function that takes an integer *n* and return all possible combinations of its factors.

**Note:**

1. You may assume that *n* is always positive.
2. Factors should be greater than 1 and less than *n*.

**Example 1:** 

```
Input: 1
Output: []
```

**Example 2:** 

```
Input: 37
Output:[]
```

**Example 3:** 

```
Input: 12
Output:
[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]
```

**Example 4:** 

```
Input: 32
Output:
[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
  [2, 4, 4],
  [4, 8]
]
```

```python
import math
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        if n <= 1: return []
        res = []
        
        def backtrack(track = [], n = n, index = 2):
            if len(track) > 0:
                res.append(track + [n])

            for i in range(index, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    backtrack(track+[i], n // i, i)

        backtrack()
        return res
```



### 320. Generalized Abbreviation

Medium

Write a function to generate the generalized abbreviations of a word. 

**Note:** The order of the output does not matter.

**Example:**

```
Input: "word"
Output:
["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
```

```python
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        
        def helper(pos, cur, count):
            if pos == len(word):
                # Once we reach the end, append current to the result
                res.append(cur + str(count) if count > 0 else cur)
            else:
                # Skip current position, and increment count
                helper(pos + 1, cur, count + 1)
                # Include current position, and zero-out count
                helper(pos + 1, cur + (str(count) if count > 0 else '') + word[pos], 0)

        res = []
        helper(0, '', 0)
        return res
```



### 811. Subdomain Visit Count

Easy

A website domain like "discuss.leetcode.com" consists of various  subdomains. At the top level, we have "com", at the next level, we have  "leetcode.com", and at the lowest level, "discuss.leetcode.com". When we visit a domain like "discuss.leetcode.com", we will also visit the  parent domains "leetcode.com" and "com" implicitly.

Now, call a "count-paired domain" to be a count (representing the  number of visits this domain received), followed by a space, followed by the address. An example of a count-paired domain might be "9001  discuss.leetcode.com".

We are given a list `cpdomains` of count-paired domains.  We would like a list of count-paired domains, (in the same format as the input, and in any order), that explicitly counts the number of visits  to each subdomain.

```
Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.
Example 2:
Input: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
Output: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
Explanation: 
We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 5 times. For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.
```

**Notes:** 

- The length of `cpdomains` will not exceed `100`. 
- The length of each domain name will not exceed `100`.
- Each address will have either 1 or 2 "." characters.
- The input count in any count-paired domain will not exceed `10000`.
- The answer output can be returned in any order.

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        if not cpdomains: return []
        
        dicty = collections.defaultdict(int)
        for item in cpdomains:
            freq = int(item.split()[0])
            url = item.split()[1]
            dicty[url] += freq
            while url:
                url = url.split('.', 1)[-1]
                dicty[url] += freq
                if '.' not in url:
                    break
        res = []
        for url, freq in dicty.items():
            res.append(str(freq) + ' ' + url)
        return res

        # string split之后是一个list    
        # string = 'ak di'
        # a, b  = string.split()
        # print(a)
        # print(b)
```



## BFS

### 107. Binary Tree Level Order Traversal II

Easy

Given a binary tree, return the *bottom-up level order* traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).

For example:
 Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```



return its bottom-up level order traversal as:

```
[
  [15,7],
  [9,20],
  [3]
]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root: return 
        
        queue = collections.deque([root])
        res = []
        while queue:
            size = len(queue)
            level = []
            for _ in range(size):
                node = queue.popleft()
                level.append(node.val)
                if node.left: 
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return reversed(res)
```



### 226. Invert Binary Tree

Easy

Invert a binary tree.

**Example:**

Input:

```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```

Output:

```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

```python
#         4
#      /    \
#     2       7
#    / \     / \
# None  3  None None
#      / \
#  None   None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        # if not root:
        #    return
        if root:
            root.left, root.right = \
            self.invertTree(root.right), self.invertTree(root.left)
        return root
```



### 404. Sum of Left Leaves

Easy

Find the sum of all left leaves in a given binary tree.

**Example:**

```
    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        return self.sumOfLeaves(root)
        
    def sumOfLeaves(self, root: TreeNode, left=False) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            if left: return root.val
            else: return 0
        return self.sumOfLeaves(root.left, True) + self.sumOfLeaves(root.right, False)

```



### 103. Binary Tree Zigzag Level Order Traversal

Medium

Given a binary tree, return the *zigzag level order* traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
 Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```



return its zigzag level order traversal as:

```
[
  [3],
  [20,9],
  [15,7]
]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return 
        
        res = []
        queue = collections.deque([root])
        reverse = False
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level += [node.val]
                if node.left: 
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if reverse == False:
                res.append(level)
            else:
                res.append(reversed(level))
            reverse = not reverse
            
        return res
```



### 199. Binary Tree Right Side View

Medium

Given a binary tree, imagine yourself standing on the *right* side of it, return the values of the nodes you can see ordered from top to bottom.

**Example:**

```
Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]
Explanation:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root: return []
        
        queue = collections.deque([root])
        res = []
        while queue:
            # level = []
            res.append(queue[-1].val)
            for _ in range(len(queue)):
                node = queue.popleft()
                # level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)      
            # res.append(level[-1])
        return res
```



### 515. Find Largest Value in Each Tree Row

Medium

Given the `root` of a binary tree, return *an array of the largest value in each row* of the tree **(0-indexed)**.

 

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/21/largest_e1.jpg)

```
Input: root = [1,3,2,5,3,null,9]
Output: [1,3,9]
```

**Example 2:**

```
Input: root = [1,2,3]
Output: [1,3]
```

**Example 3:**

```
Input: root = [1]
Output: [1]
```

**Example 4:**

```
Input: root = [1,null,2]
Output: [1,2]
```

**Example 5:**

```
Input: root = []
Output: []
```

 

**Constraints:**

- The number of the nodes in the tree will be in the range `[1, 104]`.
- `-231 <= Node.val <= 231 - 1`

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import sys
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root: return []
        res = []
        queue = collections.deque([root])
        
        while queue:
            cur_level_res = -sys.maxsize - 1
            for _ in range(len(queue)):
                node = queue.popleft()
                cur_level_res = max(cur_level_res, node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(cur_level_res)  
        return res
```



### 752. Open the Lock

Medium

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'`. The wheels can rotate freely and wrap around: for example we can turn `'9'` to be `'0'`, or `'0'` to be `'9'`. Each move consists of turning one wheel one slot.

The lock initially starts at `'0000'`, a string representing the state of the 4 wheels.

You are given a list of `deadends` dead ends, meaning if  the lock displays any of these codes, the wheels of the lock will stop  turning and you will be unable to open it.

Given a `target` representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.

 

**Example 1:**

```
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".
```

**Example 2:**

```
Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation:
We can turn the last wheel in reverse to move from "0000" -> "0009".
```

**Example 3:**

```
Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
Explanation:
We can't reach the target without getting stuck.
```

**Example 4:**

```
Input: deadends = ["0000"], target = "8888"
Output: -1
```

 

**Constraints:**

- `1 <= deadends.length <= 500`
- `deadends[i].length == 4`
- `target.length == 4`
- target **will not be** in the list `deadends`.
- `target` and `deadends[i]` consist of digits only.

```python
import collections
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        visited = set(deadends)
        queue = collections.deque(['0000'])
        depth = -1
        
        while queue:
            size = len(queue)
            depth += 1
            for _ in range(size):
                node = queue.popleft()
                if node in visited:
                    continue
                if node == target:
                    return depth
                visited.add(node)
                # queue.extend(self.successors(node))   
                queue += self.successors(node)
        return -1
    
    def successors(self, src):
        res = []
        for i, ch in enumerate(src):
            num = int(ch)
            res.append(src[:i] + str((num - 1) % 10) + src[i+1:])
            res.append(src[:i] + str((num + 1) % 10) + src[i+1:])
        return res

```



### 429. N-ary Tree Level Order Traversal

Medium

Given an n-ary tree, return the *level order* traversal of its nodes' values.

*Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See  examples).*

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

```
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]
```

 

**Constraints:**

- The height of the n-ary tree is less than or equal to `1000`
- The total number of nodes is between `[0, 10^4]`

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
import collections
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return []
        res = []
        queue = collections.deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                for child in node.children:
                    queue.append(child)
            res.append(level)
        return res
```



### 428. Serialize and Deserialize N-ary Tree

Hard

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or  transmitted across a network connection link to be reconstructed later  in the same or another computer environment.

Design an algorithm to serialize and deserialize an N-ary tree. An  N-ary tree is a rooted tree in which each node has no more than N  children. There is no restriction on how your  serialization/deserialization algorithm should work. You just need to  ensure that an N-ary tree can be serialized to a string and this string  can be deserialized to the original tree structure.

For example, you may serialize the following `3-ary` tree

![img](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

as `[1 [3[5 6] 2 4]]`. Note that this is just an example, you do not necessarily need to follow this format.

Or you can follow LeetCode's level order traversal serialization  format, where each group of children is separated by the null value.

![img](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

For example, the above tree may be serialized as `[1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]`.

You do not necessarily need to follow the above suggested formats,  there are many more different formats that work so please be creative  and come up with different approaches yourself.

 

**Constraints:**

- The height of the n-ary tree is less than or equal to `1000`
- The total number of nodes is between `[0, 10^4]`
- Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Codec:
    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.
        
        :type root: Node
        :rtype: str
        """
        if not root:
            return ''
        queue = collections.deque([root])
        res = [str(root.val)]
        
        while queue:
            node = queue.popleft()
            for child in node.children:
                queue.append(child)
                res.append(str(child.val))
            res.append('$')
        return ','.join(res)
        
    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: Node
        """
        if not data: return None
        data = data.split(',')
        root = Node(data[0])
        queue = collections.deque([root])
        
        index = 1
        while queue:
            node = queue.popleft()
            children = []
            while index < len(data):
                if data[index] != '$':
                    child = Node(data[index])
                    children.append(child)
                    queue.append(child)
                    index += 1
                else:
                    break
            node.children = children
            index += 1
        return root
       
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```



## Binary Tree

### 33. Search in Rotated Sorted Array

Medium

Given an integer array `nums` sorted in ascending order, and an integer `target`.

Suppose that `nums` is rotated at some pivot unknown to you beforehand (i.e., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`).

You should search for `target` in `nums` and if you found return its index, otherwise return `-1`.

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**Example 3:**

```
Input: nums = [1], target = 0
Output: -1
```

 \33. Search in Rotated Sorted Array

Medium

Given an integer array `nums` sorted in ascending order, and an integer `target`.

Suppose that `nums` is rotated at some pivot unknown to you beforehand (i.e., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`).

You should search for `target` in `nums` and if you found return its index, otherwise return `-1`.

 

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**Example 3:**

```
Input: nums = [1], target = 0
Output: -1
```

 ```python
# plot left, right, mid --> target senario
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums: return -1
        left, right = 0, len(nums)-1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid -1
                else:
                    left = mid + 1
            else:
                if nums[right] >= target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
 ```



### 34. Find First and Last Position of Element in Sorted Array

Medium

Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

Your algorithm's runtime complexity must be in the order of *O*(log *n*).

If the target is not found in the array, return `[-1, -1]`.

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

 

**Constraints:**

- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `nums` is a non decreasing array.
- `-10^9 <= target <= 10^9`

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums: return [-1, -1]
        
        # find left, 收右界
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                right = mid - 1
        if left >= len(nums) or nums[left] != target:
            res_left = -1
        else: res_left = left
            
            
        # find right， 收左界
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        
        if right < 0 or nums[right] != target:
            res_right = -1
        else: res_right = right
        
        return [res_left, res_right]

```



### 35. Search Insert Position

Easy

Given a sorted array and a target value, return the index if the target is  found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

**Example 1:**

```
Input: [1,3,5,6], 5
Output: 2
```

**Example 2:**

```
Input: [1,3,5,6], 2
Output: 1
```

**Example 3:**

```
Input: [1,3,5,6], 7
Output: 4
```

**Example 4:**

```
Input: [1,3,5,6], 0
Output: 0
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                right = mid -1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid -1
                    
        if left >= len(nums):
            return len(nums) 
        return left
     
        # elif left != target:
        #     return left    
        # else:
        #     return left 
```



### 50. Pow(x, n)

Medium

Implement [pow(*x*, *n*)](http://www.cplusplus.com/reference/valarray/pow/), which calculates *x* raised to the power *n* (i.e. xn).

**Example 1:**

```
Input: x = 2.00000, n = 10
Output: 1024.00000
```

**Example 2:**

```
Input: x = 2.10000, n = 3
Output: 9.26100
```

**Example 3:**

```
Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25
```

**Constraints:**

- `-100.0 < x < 100.0`
- `-231 <= n <= 231-1`
- `-104 <= xn <= 104`

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0: return 0
        if n < 0: return 1 / self.myPow(x, -n)
        if n == 0: return 1
        
        if n % 2:
            return x * self.myPow(x, n - 1)
        return self.myPow(x * x, n // 2)
```



### 81. Search in Rotated Sorted Array II

Medium

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., `[0,0,1,2,2,5,6]` might become `[2,5,6,0,0,1,2]`).

You are given a target value to search. If found in the array return `true`, otherwise return `false`.

**Example 1:**

```
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
```

**Example 2:**

```
Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if not nums: return False
        
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            
            if nums[mid] == nums[left] == nums[right]:
                left += 1
                right -= 1
               
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
                    
        return False
```



### 153. Find Minimum in Rotated Sorted Array

Medium

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  `[0,1,2,4,5,6,7]` might become  `[4,5,6,7,0,1,2]`).

Find the minimum element.

You may assume no duplicate exists in the array.

**Example 1:**

```
Input: [3,4,5,1,2] 
Output: 1
```

**Example 2:**

```
Input: [4,5,6,7,0,1,2]
Output: 0
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums: return 
        
        left, right = 0, len(nums)-1
        while left <= right: # 终止边界是 left = right + 1 （right在left）左边
            mid = left + (right - left) // 2
            if nums[mid] < nums[right]:
                right = mid 
            elif nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] == nums[right]:  # 但是走到这里就走不动了，mid和左右两边距离相等或者左边比右边小1
                break    
        return nums[left] 
```



### 154. Find Minimum in Rotated Sorted Array II

Hard

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  `[0,1,2,4,5,6,7]` might become  `[4,5,6,7,0,1,2]`).

Find the minimum element.

The array may contain duplicates.

**Example 1:**

```
Input: [1,3,5]
Output: 1
```

**Example 2:**

```
Input: [2,2,2,0,1]
Output: 0
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums: return 
        
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == nums[left] == nums[right]:
                left += 1
                right -= 1
                continue
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```



### 162. Find Peak Element

Medium

A peak element is an element that is greater than its neighbors.

Given an input array `nums`, where `nums[i] ≠ nums[i+1]`, find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that `nums[-1] = nums[n] = -∞`.

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```

**Example 2:**

```
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
```

```python
# https://leetcode-cn.com/problems/find-peak-element/solution/er-fen-fa-zhu-xing-jie-shi-python3-by-zhu_shi_fu/

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if not nums: return
        
        left, right = 0, len(nums) - 1
        while left < right:      # 终止边界： 【left = right】
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            elif nums[mid] < nums[mid + 1]:
                left = mid + 1
        return left
            
```



### 278. First Bad Version

Easy

You are a product manager and currently leading a team to develop a new  product. Unfortunately, the latest version of your product fails the  quality check. Since each version is developed based on the previous  version, all the versions after a bad version are also bad.

Suppose you have `n` versions `[1, 2, ..., n]` and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API `bool isBadVersion(version)` which will return whether `version` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

**Example:**

```
Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version. 
```

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return an integer
# def isBadVersion(version):

#相当于和两侧边界比（True：left， False：right），不用 = 情况
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid - 1
            elif not isBadVersion(mid):
                left = mid + 1
        return left
```



### 300. Longest Increasing Subsequence

Medium

Given an unsorted array of integers, find the length of longest increasing subsequence.

**Example:**

```
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
```

**Note:** 

- There may be more than one LIS combination, it is only necessary for you to return the length.
- Your algorithm should run in O(*n2*) complexity.

**Follow up:** Could you improve it to O(*n* log *n*) time complexity?

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0 
        
        def search(nums, target, left, right):
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                elif nums[mid] >= target:
                    right = mid - 1
            return left

        LIS = []
        for num in nums:
            if not LIS or num > LIS[-1]:
                LIS.append(num)
            else:
                index = search(LIS, num, 0, len(LIS)-1)
                LIS[index] = num
        return len(LIS)
```



### 540. Single Element in a Sorted Array

Medium

You are given a sorted array consisting of only integers where every  element appears exactly twice, except for one element which appears  exactly once. Find this single element that appears only once.

**Follow up:** Your solution should run in O(log n) time and O(1) space.

 

**Example 1:**

```
Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2
```

**Example 2:**

```
Input: nums = [3,3,7,7,10,11,11]
Output: 10
```

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        lo = 0
        hi = len(nums) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] == nums[mid + 1]:
                lo = mid + 2
            else:
                hi = mid
        return nums[lo]
```



### 1428. Leftmost Column with at Least a One

Medium

*(This problem is an **interactive problem**.)*

A binary matrix means that all elements are `0` or `1`. For each **individual** row of the matrix, this row is sorted in non-decreasing order.

Given a row-sorted binary matrix binaryMatrix, return leftmost column index(0-indexed) with at least a `1` in it. If such index doesn't exist, return `-1`.

**You can't access the Binary Matrix directly.** You may only access the matrix using a `BinaryMatrix` interface:

- `BinaryMatrix.get(row, col)` returns the element of the matrix at index `(row, col)` (0-indexed).
- `BinaryMatrix.dimensions()` returns a list of 2 elements `[rows, cols]`, which means the matrix is `rows * cols`.

Submissions making more than `1000` calls to `BinaryMatrix.get` will be judged *Wrong Answer*. Also, any solutions that attempt to circumvent the judge will result in disqualification.

For custom testing purposes you're given the binary matrix `mat` as input in the following four examples. You will not have access the binary matrix directly.

```python
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        m, n = binaryMatrix.dimensions()
        row, col = 0, n-1
        while row < m and col >= 0:
            if binaryMatrix.get(row, col) == 1:
                col -= 1
            else:
                row += 1
        if col != n-1:
            return col+1
        return -1
```



## Divide and Conquer

### 53. Maximum Subarray

Easy

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

**Example:**

```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

```python
import sys
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0 
        def helper(l, r):    # l, r 都是包括的
            if l > r:
                return -sys.maxsize - 1
            
            mid = (l + r) // 2
            left = helper(l, mid-1)
            right = helper(mid + 1, r)
            
            left_sum = cross_max_left = 0
            for i in reversed(range(l, mid)):
                left_sum += nums[i]
                cross_max_left = max(left_sum, cross_max_left)
            
            right_sum = cross_max_right = 0
            for i in range(mid + 1, r + 1):
                right_sum += nums[i]
                cross_max_right = max(right_sum, cross_max_right)
            
            cross = cross_max_left + cross_max_right + nums[mid]
            
            return max(left, right, cross)
            
        return helper(0, len(nums) - 1)
```



### 169. Majority Element

Easy

Given an array of size *n*, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.

You may assume that the array is non-empty and the majority element always exist in the array.

**Example 1:**

```
Input: [3,2,3]
Output: 3
```

**Example 2:**

```
Input: [2,2,1,1,1,2,2]
Output: 2
```

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int: 
        def helper(l, r):
            if l == r:
                return nums[l]

            mid = (l + r) // 2    
            left = helper(l, mid)
            right = helper(mid+1, r)

            if left == right:
                return left

            left_count = sum(1 for i in range(l, r+1) if nums[i] == left)
            right_count = sum(1 for i in range(l, r+1) if nums[i] == right)

            if left_count > right_count:
                return left
            else:
                return right
            #简化写法
            # left_count = sum(1 for i in range(lo, hi+1) if nums[i] == left)
            # right_count = sum(1 for i in range(lo, hi+1) if nums[i] == right)
            
        return helper(0, len(nums)-1)
```



### 215. Kth Largest Element in an Array

Medium

Find the **k**th largest element in an unsorted array. Note that it is the kth largest  element in the sorted order, not the kth distinct element.

**Example 1:**

```
Input: [3,2,1,5,6,4] and k = 2
Output: 5
```

**Example 2:**

```
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

**Note:** 
 You may assume k is always valid, 1 ≤ k ≤ array's length.

```python
import random
# randint(l,r), left and right are both inclusive!!

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
    
        def partition(nums, l, r):
            pivotIndex = random.randint(l, r)
            store = l
            nums[r], nums[pivotIndex] = nums[pivotIndex], nums[r]
            
            # 把所有小于or equal pivot的数放在前边
            for i in range(l, r):
                if nums[i] < nums[r]:
                    nums[i], nums[store] = nums[store], nums[i]
                    store += 1
            # 把pivot放在小于它的数的正后边
            nums[r], nums[store] = nums[store], nums[r]
            return store
        
        def quick_select(nums, l, r, k):  # 找第K小的数
            if l == r:
                return nums[l]
            p = partition(nums, l, r)
            if k == p:
                return nums[k]
            elif k < p:
                return quick_select(nums, l, p-1, k)
            else:
                return quick_select(nums, p+1, r, k)
            
        return quick_select(nums, 0, len(nums)-1, len(nums)-k)
            
```



### 235. Lowest Common Ancestor of a Binary Search Tree

Easy

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the [definition of LCA on Wikipedia](https://en.wikipedia.org/wiki/Lowest_common_ancestor): “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow **a node to be a descendant of itself**).”

Given binary search tree: root = [6,2,8,0,4,7,9,null,null,3,5]

![img](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

 

**Example 1:**

```
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
```

**Example 2:**

```
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
```

 

**Constraints:**

- All of the nodes' values will be unique.
- p and q are different and both values will exist in the BST.
