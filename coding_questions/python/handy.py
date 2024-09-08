#
import functools
from functools import cache
from typing import List
import numpy as np


#
def _eval_algo(f, *args):
    from time import time
    t0 = time()
    _ = f(*args)
    return time() - t0


def _compare_performace(*ts, labels=(), max_degree=2):
    import matplotlib.pyplot as plt
    if not labels:
        labels = list(range(len(ts)))
    fig = plt.figure(figsize=(20, 5))
    for (t, label) in zip(ts, labels):
        plt.plot(t, label=label)
    a = min([(t[2] - t[0]) / 2 for t in ts])
    for n in range(max_degree):
        plt.plot([i ** (n + 1) * a for i in range(len(ts[0]))], label=f"O(N{n + 1})")
    plt.legend()
    plt.show()


# sorts
def isSorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i+1]:
            return False
    return True


def selectionSort(arr):
    for i in range(len(arr)):
        min_ind = i
        for j in range(i + 1, len(arr)):
            if arr[min_ind] > arr[j]:
                min_ind = j
        arr[min_ind], arr[i] = arr[i], arr[min_ind]


def insertionSort(arr):
    for i in range(1, len(arr)):
        cur = arr[i]
        j = i
        while j > 0 and arr[j-1] > cur:
            arr[j] = arr[j-1]
            j -= 1
        arr[j] = cur


def bubbleSort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j-1]:
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1


def bubbleSort(arr):
    for i in range(len(arr)-1, 0, -1):
        swapped = False
        for j in range(i):
            if arr[j+1] < arr[j]:
                arr[j+1], arr[j] = arr[j], arr[j+1]
                swapped = True
        if not swapped:
            return


def mergeSort(arr):
    def _mergeSort(i, j):
        if i >= j:
            return
        m = (i + j) // 2
        _mergeSort(i, m)
        _mergeSort(m+1, j)
        l, r = i, m+1
        aux[i:j+1] = arr[i:j+1]
        for k in range(i, j+1):
            if l > m:
                arr[k] = aux[r]
                r += 1
            elif r > j:
                arr[k] = aux[l]
                l += 1
            elif aux[l] < aux[r]:
                arr[k] = aux[l]
                l += 1
            else:
                arr[k] = aux[r]
                r += 1
    aux = [d for d in arr]
    _mergeSort(0, len(arr) - 1)


def quickSort(arr):

    def _partition(i, j):
        last = i + 1
        for k in range(i+1, j+1):
            if arr[k] < arr[i]:
                arr[k], arr[last] = arr[last], arr[k]
                last += 1
        arr[i], arr[last - 1] = arr[last - 1], arr[i]
        return last - 1

    def _partition2(i, j):
        l, r = i+1, j
        while l <= r:
            if arr[l] <= arr[i]:
                l += 1
            else:
                arr[r], arr[l] = arr[l], arr[r]
                r -= 1
        l -= 1
        arr[i], arr[l] = arr[l], arr[r]
        return l

    def _quickSort(i, j):
        if i >= j:
            return
        pi = _partition(i, j)
        _quickSort(i, pi - 1)
        _quickSort(pi + 1, j)

    _quickSort(0, len(arr) - 1)


# bucket sort
# https://leetcode.com/problems/maximum-gap/


# array    
def copyArrayElems(arr, i, j, l):
    if i == j:
        return 
    
    n = len(arr)
    if j < i:
        for k in range(min(n - i, l)):
            arr[j + k] = arr[i + k]
    else:
        cap = j - i
        for k in range(min(n - j, l)):
            arr[j + k], arr[i + k % cap] = arr[i + k % cap], arr[j + k]
        for k_ in range(min([cap, l, n-j])):
            arr[i + k_] = arr[j + k_]


def copyArrayElems2(arr, i, j, l):
    """
    Copy l array elements from position i to position j without using extra memory.
    
    Parameters:
    arr (list): The input array
    i (int): The starting index of the elements to be copied
    j (int): The destination index where elements will be copied
    l (int): The number of elements to copy
    
    Returns:
    None
    """
    # Determine the direction of copying based on the relative position of i and j
    n = len(arr)
    if j > i:  # If copying forward, go backwards to avoid overwriting
        for k in range(min(n - j, l)-1, -1, -1):
            arr[j+k] = arr[i+k]
    else:  # If copying backward, go forward
        for k in range(min(n-i, l)):
            arr[j+k] = arr[i+k]


def removeDuplicates(nums):
    """
    Input: nums = [1,1,1,2,2,3]
    Output: 5, nums = [1,1,2,2,3,_]
    Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    """
    n = len(nums)
    i = 1
    cnt = 1
    for j in range(1, n):
        cnt = 0 if nums[j] != nums[j-1] else (cnt + 1)
        if cnt <= 2:
            nums[i] = nums[j]
            i += 1
    return i


def rotateLeft(arr, k):
    def _reverse(i, j):
        while i < j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
    n = len(arr)
    k %= n 
    if k == 0:
        return
    _reverse(0, k-1)
    _reverse(k, n-1)
    _reverse(0, n-1)


def rotateRight(arr, k):
    def _reverse(i, j):
        while i < j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
    n = len(arr)
    k = k % n
    if k == 0:
        return
    _reverse(0, n-1)
    _reverse(0, k-1)
    _reverse(k, n-1)


def countSmaller(arr):
    # find numbers of elements smaller on the right
    def _merge(l, m, r):
        i, j = l, m + 1
        for k in range(l, r+1):
            if i > m or (j <= r and arr[inds[j]] < arr[inds[i]]):
                inds_aux[k] = inds[j]
                j += 1
            else:
                inds_aux[k] = inds[i]
                cnts[inds[i]] += j - m - 1
                i += 1
        for k in range(l, r+1):
            inds[k] = inds_aux[k]

    def _mergeSort(l, r):
        if l >= r:
            return
        m = (l + r) // 2
        _mergeSort(l, m)
        _mergeSort(m+1, r)
        _merge(l, m, r)

    n = len(arr)
    inds = list(range(n))
    inds_aux = inds.copy()
    cnts = [0] * n
    _mergeSort(0, n-1)

    return cnts


def majorityElement(arr):
    cnt = 0
    maj_ind = 0
    n = len(arr)
    for i in range(n):
        cnt += 1 if arr[i] == arr[maj_ind] else -1
        if cnt == 0:
            cnt = 1
            maj_ind = i
    return arr[maj_ind] if sum([a == arr[maj_ind] for a in arr]) > n // 2 else -1


def circular_tour_patrolpump(lis, n):
    """
    Input:
    N = 4
    Petrol = 4 6 7 4
    Distance = 6 5 3 5
    Output: 1
    Explanation: There are 4 petrol pumps with
    amount of petrol and distance to next
    petrol pump value pairs as {4, 6}, {6, 5},
    {7, 3} and {4, 5}. The first point from
    where truck can make a circular tour is
    2nd petrol pump. Output in this case is 1
    (index of 2nd petrol pump).
    """
    diffs = [t[0] - t[1] for t in lis]
    if sum(diffs) < 0:
        return -1
    i = 0
    total = 0
    start = 0
    while i < n:
        total += diffs[i]
        if total < 0:
            start = i + 1
            total = 0
        i += 1
    return start


def findPages(arr, n, m):
    """
    N = 4
    A[] = {12,34,67,90}
    M = 2
    Output:113
    Explanation:Allocation can be done in
    following ways:{12} and {34, 67, 90}
    Maximum Pages = 191{12, 34} and {67, 90}
    Maximum Pages = 157{12, 34, 67} and {90}
    Maximum Pages =113. Therefore, the minimum
    of these cases is 113, which is selected
    as the output.
    """
    def _get_num_bs(l):
        n_b = 1
        total = 0
        for a in arr:
            if total + a > l:
                n_b += 1
                total = 0
            total += a
        return n_b

    l, r = min(arr), sum(arr)
    while l <= r:
        l_b = (l + r) // 2
        n_b = _get_num_bs(l_b)
        if n_b > m:
            l = l_b + 1
        else:
            r = l_b - 1    

    return l if n_b >= m else l - 1


def canReach(A, N):
    # if it can reach end of arr
    i = 0
    max_i = 0
    while i < N:
        max_i = max(i + A[i], max_i)
        if max_i >= N - 1:
            return True
        if i >= max_i:
            break
        i += 1
    return False


def minJumpsToEnd(arr):
    n = len(arr)
    cur_reach = 0
    max_reach = 0
    jumps = 0
    for i, a in enumerate(arr):
        max_reach = max(max_reach, i + a)
        if max_reach >= n - 1:
            return jumps + 1
        if i == cur_reach:
            if cur_reach == max_reach:
                return float('inf')
            cur_reach = max_reach
            jumps += 1
    return float('inf')


def findMedSortedArrays(arr1, n1, arr2, n2):
    """
    Finds the median of values stored in two sorted arrays using binary search.

    :param arr1: First sorted array.
    :param n1: Length of the first array.
    :param arr2: Second sorted array.
    :param n2: Length of the second array.
    :return: The median of the two sorted arrays.
    """
    # Ensure arr1 is the smaller array for efficient binary search
    if n1 > n2:
        return findMedSortedArrays(arr2, n2, arr1, n1)
    
    low, high = 0, n1
    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (n1 + n2 + 1) // 2 - partitionX
        
        # If partitionX is 0 it means nothing is there on left side. Use -inf for maxLeftX
        maxLeftX = float('-inf') if partitionX == 0 else arr1[partitionX - 1]
        # If partitionX is length of array it means nothing is there on right side. Use +inf for minRightX
        minRightX = float('inf') if partitionX == n1 else arr1[partitionX]
        
        maxLeftY = float('-inf') if partitionY == 0 else arr2[partitionY - 1]
        minRightY = float('inf') if partitionY == n2 else arr2[partitionY]
        
        if maxLeftX <= minRightY and maxLeftY <= minRightX:
            # Check if total length is even
            if (n1 + n2) % 2 == 0:
                return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
            else:
                return max(maxLeftX, maxLeftY)
        elif maxLeftX > minRightY:
            high = partitionX - 1
        else:
            low = partitionX + 1


def kadane(arr):
    max_sum = float("-inf")
    cur_sum = 0
    n = len(arr)
    l = r = 0
    i = j = 0
    while j < n:
        cur_sum += arr[j]
        if cur_sum > max_sum:
            max_sum = cur_sum
            l, r = i, j
        j += 1
        if cur_sum <= 0:
            cur_sum = 0
            i = j
    return max_sum


def kdn(arr):
    max_sum = float("-inf")
    cur_sum = 0
    for a in arr:
        cur_sum = max(cur_sum, 0) + a
        max_sum = max(max_sum, cur_sum)
    return max_sum


def kdnCircularArr(arr):
    # https://leetcode.com/problems/maximum-sum-circular-subarray/discuss/178422/One-Pass
    total = cur_max = cur_min = 0
    max_sum, min_sum = arr[0], arr[0]
    for a in arr:
        total += a
        cur_max = max(cur_max, 0) + a
        max_sum = max(max_sum, cur_max)
        cur_min = min(cur_min, 0) + a
        min_sum = min(min_sum, cur_min)

    return max(max_sum, total - min_sum) if max_sum > 0 else max_sum


def maxNumber(nums1: List[int], nums2: List[int], k: int) -> List[int]:
    def _merge_max(a, b):
        return [max(a,b).pop(0) for _ in a+b]
    def _topK_stable(a, k):
        if k == 0:
            return []
        stk = []
        max_del = len(a) - k
        for d in a:
            while stk and max_del and stk[-1] < d:
                stk.pop()
                max_del -= 1
            stk.append(d)
        return stk[:k]
    return max(_merge_max(_topK_stable(nums1, i), _topK_stable(nums2, k-i))
               for i in range(k+1) if i <= len(nums1) and k-i <= len(nums2))


def one_to_ten_count(arr):
    assert len(arr) >= 10, "array should be at least size of 10 (1 ... 10 values)"
    # manipulate numbers to use them as counters for 1 to 10 for the first 10 numbers
    for i in range(10):
        arr[i] *= 100
    for i in range(10):
        arr[arr[i]//100-1] += 1
    for i in range(10):
        arr[i] %= 100
    for i in range(10, len(arr)):
        arr[arr[i] - 1] += 1
    return arr[:10]


def exec_time(arr, cool):
    """
    Computes the total time to execute the tasks with a cooldown interval
    cooldown interval is the time to wait before executing the same task again
    Tasks: 1, 2, 3, 1, 2, 3
    Recovery interval (cooldown): 3
    1 2 3 - 1 2 3
    answer: 7
    """
    def sol_1_q(arr, cool): 
        q = [None] * cool
        total = 0
        for a in arr:
            if a in q:
                while q.pop(0) != a:
                    q.append(None)
                    total += 1
                q.append(None)
                total += 1
            q.pop(0)
            q.append(a)
            total += 1
        return total

    def sol_2_q(arr, cool):
        q = []
        in_q = set()
        t_total = 0
        for a in arr:
            next_t = t_total
            while q and (q[0][1] < t_total or a in in_q):
                a_, next_t = q.pop(0)
                in_q.remove(a_)
            t_total = max(t_total, next_t)
            t_total += 1
            q.append((a, t_total + cool))
            in_q.add(a)
        return t_total    

    def sol_cache(arr, cool):
        next_valid = {}
        t_total = 0
        for a in arr:
            next_t = next_valid.get(a, 0)
            t_total = max(t_total, next_t)
            t_total += 1
            next_valid[a] = t_total + cool
        return t_total


def minLenMaxSum(arr):
    def _minLenMaxSumStk():
        cum_sums = [0] + np.cumsum(arr).tolist()
        max_sum = float('-inf')
        n = len(arr)
        l, r = 0, n
        stk = [0]
        for j in range(1, n+1):
            while stk and cum_sums[stk[-1]] >= cum_sums[j]:
                stk.pop()
            i = j - 1 if not stk else stk[0]
            cur_sum = cum_sums[j] - cum_sums[i]
            if cur_sum > max_sum or (cur_sum == max_sum and r - l > j - i):
                max_sum = cur_sum
                l, r = i, j
            stk.append(j)
        return max_sum, l, r

    def _minLenMaxSumKadanes(arr):
        cur_sum = 0
        max_sum = float('-inf')
        l, r = 0, len(arr)
        i = 0
        for j, a in enumerate(arr):
            cur_sum = max(cur_sum, 0) + a
            if max_sum < cur_sum or (max_sum == cur_sum and j - i < r - l):
                max_sum = cur_sum
                l, r = i, j
            if cur_sum <= 0:
                i = j + 1
        return max_sum, l, r+1
    
    return _minLenMaxSumKadanes(arr)


def maxLenMaxSum(arr):
    def _maxLenMaxSumStk():
        cum_sum = [0] + arr[:]
        n = len(cum_sum)
        for i in range(1, n):
            cum_sum[i] += cum_sum[i-1]
        stk = [0]
        l, r = 0, 0
        max_sum = float('-inf')
        for j in range(1, n):
            while stk and cum_sum[stk[-1]] > cum_sum[j]:
                stk.pop()
            i = stk[0] if stk else j - 1
            cur_sum = cum_sum[j] - cum_sum[i]
            if cur_sum > max_sum or (cur_sum == max_sum and j - i > r - l):
                max_sum, l, r = cur_sum, i,  j
            stk.append(j)
        return max_sum, l, r

    def _maxLenMaxSumKadane():
        n = len(arr)
        i, j = 0, 0
        l, r = 0, 0
        max_sum = float('-inf')
        total = 0
        while j < n:
            if total < 0:
                total -= arr[i]
                i += 1
            else:
                total += arr[j]
                j += 1
                if total > max_sum or (total == max_sum and j - i > r - l):
                    max_sum, l, r = total, i, j
        return max_sum, l, r

    return _maxLenMaxSumKadane()


def minLenSumTarget(arr, target):
    total = 0
    l, r = 0, float('inf')
    seen = {0: -1}
    for j, a in enumerate(arr):
        total += a
        if total - target in seen:
            i = seen[total - target]
            if j - i < r - l:
                l, r = i, j
        seen[total] = j
    return (l+1, r+1) if r < float('inf') else (-1, -1)


def maxLenSumTarget(arr, target):
    total = 0
    l, r = -1, -1
    seen = {0: -1}
    for j, a in enumerate(arr):
        total += a
        if total - target in seen:
            i = seen[total - target]
            l, r = i, j
        if total not in seen:
            seen[total] = j
    return (l+1, r+1) if r > -1 else (-1, -1)


def countRangeSum(nums: List[int], lower: int, upper: int) -> int:
    # numbers of ranges sums between lower and upper
    from bisect import bisect_left, bisect_right
    total = 0
    cml_sums_srtd = [0]
    cnt = 0
    for a in nums:
        total += a
        l = total - upper
        r = total - lower
        i, j = bisect_left(cml_sums_srtd, l), bisect_right(cml_sums_srtd, r)
        cnt += (j - i)
        ind = bisect_left(cml_sums_srtd, total)
        cml_sums_srtd.insert(ind, total)
    return cnt


def maxArea(arr):
    stk = []
    max_area = 0
    n = len(arr)
    for i in range(len(arr)):
        while stk and arr[stk[-1]] >= arr[i]:
            ind = stk.pop()
            j = -1 if not stk else stk[-1]
            max_area = max(max_area, (i - j - 1) * arr[ind])
        stk.append(i)
    while stk:
        ind = stk.pop()
        j = -1 if not stk else stk[-1]
        max_area = max(max_area, (n - j - 1) * arr[ind])
    return max_area


def totalVolumeArrayTraverseSoln(arr):
    maxes_left = [0] + arr[1:]
    maxes_right = arr[:-1] + [0]
    for i in range(1, len(arr)):
        maxes_left[i] = max(maxes_left[i-1], arr[i-1])
    for i in range(len(arr) - 2, -1, -1):
        maxes_right[i] = max(maxes_right[i+1], arr[i+1])
    total_vol = 0
    for i in range(1, len(arr) - 1):
        min_edge = min(maxes_left[i], maxes_right[i])
        total_vol += (max(min_edge, arr[i]) - arr[i])
    return total_vol


def maxVolumeArrayTraverseSoln(arr):
    maxes_left = [0] + arr[1:]
    maxes_right = arr[:-1] + [0]
    for i in range(1, len(arr)):
        maxes_left[i] = max(maxes_left[i-1], arr[i-1])
    for i in range(len(arr) - 2, -1, -1):
        maxes_right[i] = max(maxes_right[i+1], arr[i+1])
    max_vol = 0
    vol = 0
    for i in range(1, len(arr) - 1):
        min_edge = min(maxes_left[i], maxes_right[i])
        if min_edge <= arr[i]:
            max_vol = max(max_vol, vol)
            vol = 0
            continue
        vol += (min_edge - arr[i])
    return max(max_vol, vol)


def minimumDeletions(s):
    """
    https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/
    Input: s = "aababbab"
    Output: 2
    Explanation: You can either:
    Delete the characters at 0-indexed positions 2 and 6 ("aababbab" -> "aaabbb"), or
    Delete the characters at 0-indexed positions 3 and 6 ("aababbab" -> "aabbbb").
    """
    a, b = s.count('a'), 0
    res = a + b
    for char in s:
        if char == 'a':
            a -= 1
        else:
            b += 1
        res = min(res, a + b)
    return res


def containsNearbyAlmostDuplicate(nums: List[int], k: int, t: int) -> bool:
    """
    Given an integer array nums and two integers k and t,
    return true if there are two distinct indices i and j in the array such that abs(nums[i] - nums[j])
    <= t and abs(i - j) <= k.
    """
    buckets = {}
    for i, num in enumerate(nums):
        b = num // (t + 1)
        if b in buckets or (b - 1 in buckets and abs(buckets[b - 1] - num) <= t) or (
                b + 1 in buckets and abs(num - buckets[b + 1]) <= t):
            return True
        buckets[b] = num
        if i >= k:
            del buckets[nums[i - k] // (t + 1)]
    return False


def fourSumCount(target, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
    d = {}
    r = 0
    for i in range(len(nums1)):
        for j in range(len(nums2)):
            d[nums1[i] + nums2[j]] = d.get(nums1[i] + nums2[j], 0) + 1

    for i in range(len(nums3)):
        for j in range(len(nums4)):
            r += d.get(target-(nums3[i]+nums4[j]), 0)
    return r


def minimumPlatform(n, arr, dep):
    # minimum platforms needed arrival departures
    arr_dep = sorted([(a, 'a') for a in arr] + [(d, 'd') for d in dep])
    cnt = 0
    min_cnt = 1
    print(arr_dep)
    for t in arr_dep:
        if t[1] == 'a':
            cnt += 1
        else:
            cnt -= 1
        min_cnt = max(min_cnt, cnt)
    return min_cnt


def fourSumUniq_On3(arr, total):
    arr.sort()
    n = len(arr)
    results = []
    i = 0
    while i < n-3:
        j = i + 1
        while j < n - 2:
            l, r = j+1, n-1
            while l < r:
                cur_sum = arr[l] + arr[r] + arr[i] + arr[j]
                if cur_sum == total:
                    results.append((arr[i], arr[j], arr[l], arr[r]))
                    while l < r - 1 and arr[l] == arr[l+1]:
                        l += 1
                    l += 1
                elif cur_sum > total:
                    r -= 1
                else:
                    l += 1
            while j < n - 3 and arr[j] == arr[j+1]:
                j += 1
            j += 1
        while i < n - 1 and arr[i] == arr[i+1]:
            i += 1
        i += 1
    return results


def fourSumUniq_On2(arr, target):
    n = len(arr)
    results = set()
    
    seen_1_2 = {}
    for i in range(n-1):
        for j in range(i+1, n):
            cur_sum = arr[i]+arr[j]
            seen_1_2[cur_sum] = (seen_1_2.get(cur_sum) or set())
            seen_1_2[cur_sum].add((i, j))
            
    for i in range(n-1):
        for j in range(i+1, n):
            cur_sum = arr[i] + arr[j]
            seen = seen_1_2.get(target - cur_sum)
            if seen is not None:
                for t in seen:
                    if i not in t and j not in t:
                        cur_slct = sorted([arr[k] for k in t + (i, j)])
                        results.add(tuple(cur_slct))

    return [t for t in results]


def findKthNumber(n, k):
    """Given two integers n and k, return the kth lexicographically smallest integer in the range [1, n].
    Example 1:

    Input: n = 13, k = 2
    Output: 10
    Explanation: The lexicographical order is [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9], so the second smallest number
    is 10.
    """
    base = 1
    k -= 1
    while k > 0:
        count = 0
        span = [base, base+1]
        while span[0] <= n:
            count += min(n+1, span[1]) - span[0]
            span = [10*span[0], 10*span[1]]
        if k >= count:
            k -= count
            base += 1
        else:
            k -= 1
            base *= 10
    return base


def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """
    Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    Output: [[1,5],[6,9]]
    """
    from bisect import bisect_left, bisect_right
    l = bisect_left([t[1] for t in intervals], newInterval[0])
    r = bisect_right([t[0] for t in intervals], newInterval[1])
    if l == r:
        return intervals[:l] + [newInterval] + intervals[l:]
    return intervals[:l] + [(min(intervals[l][0], newInterval[0]), max(intervals[r-1][1], newInterval[1]))] + intervals[r:]        


# stack, priority queue
def subsets(arr):
    def _calcSubset_rec(i, cur_sub):
        if i == n:
            subsets.append(cur_sub)
            return
        _calcSubset_rec(i+1, cur_sub)
        _calcSubset_rec(i+1, cur_sub + [arr[i]])
    
    def _calcSubset_Stack_rec(index):
        subsets.append(stk[:])    
        for i in range(index, n):            
            stk.append(arr[i])
            _calcSubset_Stack_rec(i + 1)
            _ = stk.pop()

    def _calcSubset_Queue(arr):
        subsets = []
        q = [[]]
        n = len(arr)
        while q:
            t = q.pop(0)
            subsets.append([arr[i] for i in t])
            ind = 0 if not t else t[-1] + 1
            for i in range(ind, n):
                q.append(t + [i])   
        return subsets
        
    subsets, stk = [], []
    n = len(arr)
    _calcSubset_Stack_rec(0)

    return subsets


def longestValidParentheses(s):
    stk = []
    for c in s:
        if c == "(":
            stk.append(c)
            continue
        cur = 0
        while stk and isinstance(stk[-1], int):
            cur += stk.pop()
        if stk and stk[-1] == "(":
            stk.pop()
            cur += 2
            stk.append(cur)
        else:
            stk.append(cur)
            stk.append(c)
    max_len = 0
    cur = 0
    while stk:
        d = stk.pop()
        if not isinstance(d, int):
            cur = 0
        else:
            cur += d
            max_len = max(max_len, cur)
    return max_len


def minOperationsToFlip(s):
    """
    https://leetcode.com/problems/minimum-cost-to-change-the-final-value-of-expression/
    Input: expression = "1&(0|1)"
    Output: 1
    Explanation: We can turn "1&(0|1)" into "1&(0&1)" by changing the '|' to a '&' using 1 operation.
    The new expression evaluates to 0.
    """
    jump = {}
    stack = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            jump[i] = stack.pop()

    def change(i, j):
        if jump.get(j) == i:
            return change(i + 1, j - 1)
        if i == j:
            return int(s[i]), 1
        k = jump[j] if j in jump else j
        l, cl = change(i, k - 2)
        r, cr = change(k, j)
        return (l & r if s[k - 1] == '&' else l | r,
                1 if l != r else min(cl, cr) + (l == 1 and s[k - 1] == '|' or l == 0 and s[k - 1] == '&'))

    def stk_soln(s):
        
        def _merge(t1, t2, op):
            return (eval(f"{t1[0]}{op}{t2[0]}"),
                    1 if t1[0] != t2[0] else min(t1[1], t2[1]) + ((op == "&" and t1[0] == 0) or (op == "|" and t1[0] == 1)))
        
        stk = []
        ops = ["&", "|"]
        for c in s:
            if c == "(" or c in ops:
                stk.append(c)
                continue
            if c == ")":
                t2 = stk.pop()
                assert stk.pop() == "(", "mistake"
            else:
                t2 = (int(c), 1)
            if stk and stk[-1] in ops:
                op = stk.pop()
                t1 = stk.pop()
                t2 = _merge(t1, t2, op)
            stk.append(t2)
        return stk[-1]

    return change(0, len(s) - 1)[1]


def medianOfStream(arr):
    import heapq
    a, b = min(arr[:2]), max(arr[:2])
    min_heap = [b]
    max_heap = [-a]
    med = (a + b) / 2
    for d in arr[2:]:
        if d < med:
            if len(max_heap) > len(min_heap):
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
            heapq.heappush(max_heap, -d)
        else:
            if len(min_heap) > len(max_heap):
                heapq.heappush(max_heap, -heapq.heappop(min_heap))
            heapq.heappush(min_heap, d)
        if len(min_heap) > len(max_heap):
            med = min_heap[0]
        elif len(max_heap) > len(min_heap):
            med = -max_heap[0]
        else:
            med = (min_heap[0] - max_heap[0]) / 2
    return med


def maxSlidingWindow(nums, k):
    stk = []
    maxes = []
    for i, d in enumerate(nums):
        while stk and stk[0][0] <= i - k:
            stk.pop(0)
        while stk and stk[-1][1] <= d:
            stk.pop()
        stk.append((i, d))
        if i < k - 1:
            continue
        maxes.append(stk[0][1])
    return maxes


def medianSlidingWindow(nms, k):
    pass


def scheduleCourse(courses: List[List[int]]) -> int:
    """
    Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
    Output: 3
    Explanation:
    There are totally 4 courses, but you can take 3 courses at most:
    First, take the 1st course, it costs 100 days so you will finish it on the 100th day, and ready to take the next course on the 101st day.
    Second, take the 3rd course, it costs 1000 days so you will finish it on the 1100th day, and ready to take the next course on the 1101st day.
    Third, take the 2nd course, it costs 200 days so you will finish it on the 1300th day.
    The 4th course cannot be taken now, since you will finish it on the 3300th day, which exceeds the closed date.
    """
    import heapq
    t = 0
    pq = []
    for dt, te in sorted(courses, key=lambda x: x[1]):
        t += dt
        heapq.heappush(pq, -dt)
        if t > te:  # make current schedule valid again, and t is minimum
            t += heapq.heappop(pq)
    return len(pq)


def getSkyline(buildings: List[List[int]]) -> List[List[int]]:
    class Interval:
        def __init__(self, l, r, h):
            self.l = l
            self.r = r
            self.h = h

        def __lt__(self, other):
            return self.h > other.h

    import heapq
    l = buildings + [(t[1], 0, 0) for t in buildings]
    l.sort(key=lambda t: (t[0], -t[2]))
    intervals = [Interval(*t) for t in l]
    pq = []
    skyline = []
    for i in intervals:
        while pq and pq[0].r <= i.l:
            heapq.heappop(pq)
        heapq.heappush(pq, i)
        if not skyline or skyline[-1][1] != pq[0].h:
            skyline.append((i.l, pq[0].h))
    return skyline


class StackMinO1:
    def __init__(self):
        self.stk = []
        self.minEl = None

    def push(self, x):
        if not self.stk:
            self.stk.append(x)
            self.minEl = x
            return
        if x < self.minEl:
            x, self.minEl = 2 * x - self.minEl, x
        self.stk.append(x)

    def pop(self):
        if not self.stk:
            return
        val = self.stk.pop()
        if not self.stk:
            self.minEl = None
        elif val < self.minEl:
            val, self.minEl = self.minEl, 2 * self.minEl - val
        return val

    def getMin(self):
        return self.minEl


# linked lists
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


def has_cycle(ll):
    def _bruteforce():
        nodes = set()
        while ll:
            if ll in nodes:
                return True
            nodes.add(ll)
            ll = ll.next
    def _fast_slow(ll):
        fast = ll
        slow = ll
        while fast:
            fast = fast.next
            if not fast:
                return False
            fast = fast.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    return _fast_slow(ll)


def ll_to_arr(ll):
    nodes = set()
    arr = []
    while ll:
        if ll in nodes:
            print(f"cycle! arr: {arr}")
            break
        nodes.add(ll)
        arr.append(ll.data)
        ll = ll.next
    return arr


def arr_to_ll(arr):
    head = Node()
    node = head
    for a in arr:
        node.next = Node(a)
        node = node.next
    return head.next


def quickSortLL_dataswap(ll):
    def _partition(l, r):
        p = l
        cur = p
        l = l.next
        while l != r:
            if l.data <= p.data:
                cur = cur.next
                l.data, cur.data = cur.data, l.data
            l = l.next
        cur.data, p.data = p.data, cur.data
        p = cur
        return p

    def _quickSort(l, r):
        if l is None or l.next == r or l == r:
            return
        p = _partition(l, r)
        _quickSort(l, p)
        _quickSort(p.next, r)

    _quickSort(ll, None)


def quickSortLL(ll):
    def _partition(l, r):

        head_l, head_r = Node(), Node()
        tail_l = head_l
        tail_r = head_r
        p = l
        l = l.next
        while l != r:
            if l.data <= p.data:
                tail_l.next = l
                tail_l = tail_l.next
            else:
                tail_r.next = l
                tail_r = tail_r.next
            l = l.next

        tail_r.next = r
        head_r = head_r.next
        p.next = head_r
        tail_l.next = p
        head_l = head_l.next

        return head_l, p

    def _quickSort(l, r):
        if l == r:
            return l
        l, p = _partition(l, r), l
        head_l = _quickSort(l, p)
        head_r = _quickSort(p.next, r)
        p.next = head_r

        return head_l

    return _quickSort(ll, None)


def mergeSortLL(ll):
    def _merge(head_l, head_r):
        head = tail = Node()
        while head_l or head_r:
            if not head_l or (head_r and head_r.data < head_l.data):
                tail.next = head_r
                head_r = head_r.next
            else:
                tail.next = head_l
                head_l = head_l.next
            tail = tail.next
        return head.next

    def _mergeSort(l, n):
        if n == 1:
            l.next = None
            return l

        m = l
        for i in range(n // 2):
            m = m.next

        head_l = _mergeSort(l, n//2)
        head_r = _mergeSort(m, n - n // 2)
        head = _merge(head_l, head_r)

        return head

    n = 0
    head = ll
    while head:
        n += 1
        head = head.next

    return _mergeSort(ll, n)


class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.l = None
        self.r = None


class LRUCache:
    """Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    Implement the LRUCache class:
    LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    int get(int key) Return the value of the key if the key exists, otherwise return -1.
    void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair
    to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
    The functions get and put must each run in O(1) average time complexity.
    """
    def __init__(self, capacity):
        self.head = None
        self.tail = None
        self.capacity = capacity
        self.keys = {}

    def get(self, key):
        if key in self.keys:
            self._update(key)
            return self.keys[key].val
        return -1

    def put(self, key, value):
        if key in self.keys:
            self.keys[key].val = value
            self._update(key)
        else:
            node = Node(key, value)
            self.keys[key] = node
            if self.head is None:
                self.head = node
                self.tail = node
            else:
                prev_head, self.head = self.head, node
                prev_head.r, self.head.l = self.head, prev_head
            if len(self.keys) > self.capacity:
                self._deltail()

    def _update(self, key):
        node = self.keys[key]
        if node == self.head:
            return
        node.r.l = node.l
        if node == self.tail:
            self.tail = node.r
        if node.l:
            node.l.r = node.r
        prev_head, self.head = self.head, node
        prev_head.r, self.head.l = self.head, prev_head

    def _deltail(self):
        tail, self.tail = self.tail, self.tail.r
        del self.keys[tail.key]
        del tail
        self.tail.l = None


def removeLoopLL(ll):
    fast = slow = ll

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            break

    if not (fast and fast.next):
        return

    slow = ll
    while slow != fast:
        slow = slow.next
        fast = fast.next

    while fast.next != slow:
        fast = fast.next

    fast.next = None


# https://leetcode.com/problems/find-the-duplicate-number/submissions/


# union find
class UF:
    # quick union
    def __init__(self, n):
        self.ids = [i for i in range(n)]
        self.cnt = n
    def find(self, i):
        while i != self.ids[i]:
            i = self.ids[i]
        return i
    def connected(self, i, j):
        return self.find(i) == self.find(j)
    def union(self, i, j):
        i_id = self.find(i)
        j_id = self.find(j)
        self.ids[i_id] = self.ids[j_id]
        if i_id != j_id:
            self.cnt -= 1


class WeightedQUF(UF):
    # weigthed quick union find with path compression
    def __init__(self, n):
        super(WeightedQUF, self).__init__(n)
        self.weights = [1 for _ in range(n)]
    def find(self, i):
        while i != self.ids[i]:
            i_ = i
            i = self.ids[i]
            self.ids[i_] = i
        return i
    def union(self, i, j):
        i_id = self.ids[i]
        j_id = self.ids[j]
        if i_id == j_id:
            return
        if self.weights[i_id] > self.weights[j_id]:
            self.ids[j_id] = i_id
            self.weights[i_id] += self.weights[j_id]
        else:
            self.ids[i_id] = j_id
            self.weights[j_id] += self.weights[i_id]
        if i_id != j_id:
            self.cnt -= 1


# trees
class IndexMinPQ:
    def __init__(self, n):
        self.n = 0
        self.pq = [None] * (n + 1)
        self.qp = [None] * (n + 1)
        self.vals = [None] * (n + 1)

    def put(self, ind, val):
        if self.vals[ind] is None:
            self.vals[ind] = val
            self.n += 1
            self.qp[ind] = self.n
            self.pq[self.n] = ind
            self._swim(self.qp[ind])
        else:
            cur_val = self.vals[ind]
            if cur_val == val:
                return
            self.vals[ind] = val
            if val < cur_val:
                self._swim(self.qp[ind])
            else:
                self._sink(self.qp[ind])

    def pop(self):
        if self.n == 0:
            return
        ind, val = self.pq[1], self.vals[self.pq[1]]
        self.pq[1] = self.pq[self.n]
        self.qp[self.pq[1]] = 1
        self.qp[ind], self.vals[ind] = None, None
        self.pq[self.n] = None
        self.n -= 1
        if self.n > 0:
            self._sink(1)
        return ind, val

    def _sink(self, pos):
        while pos * 2 <= self.n:
            pos_ = pos * 2
            if pos_ + 1 <= self.n and self.vals[self.pq[pos_+1]] < self.vals[self.pq[pos_]]:
                pos_ += 1
            if self.vals[self.pq[pos_]] >= self.vals[self.pq[pos]]:
                return
            self.pq[pos], self.pq[pos_] = self.pq[pos_], self.pq[pos]
            self.qp[self.pq[pos]], self.qp[self.pq[pos_]] = pos, pos_
            pos = pos_

    def _swim(self, pos):
        while pos > 1:
            pos_ = pos // 2
            if self.vals[self.pq[pos]] >= self.vals[self.pq[pos_]]:
                return
            self.pq[pos], self.pq[pos_] = self.pq[pos_], self.pq[pos]
            self.qp[self.pq[pos]], self.qp[self.pq[pos_]] = pos, pos_
            pos = pos_


def get_adjs_from_weigthed_adjlist(weigthed_adj_list, n):
    adjs = {i: set() for i in range(n)}
    for t in weigthed_adj_list:
        adjs[t[1]].add((t[0], t[2]))
        adjs[t[2]].add((t[0], t[1]))
    return adjs


def minSpanningTreeLazyPrim(adjs_weighted, n):
    def _visit(p):
        visited[p] = True
        for w, q in adjs_weighted[p]:
            if not visited[q]:
                heapq.heappush(pq, (w, p, q))
    import heapq
    visited = [False] * n
    pq = []
    tree = []
    for s in adjs_weighted:
        if not visited[s]:
            _visit(s)
        while pq and len(tree) < n - 1:
            w, i, j = heapq.heappop(pq)
            if not visited[j]:
                tree.append((min(i, j), max(i, j), w))
                _visit(j)
    return sorted(tree)


def minSpanningTreeEagerPrim(adjs_weighted, n):
    def _visit(p):
        visited[p] = True
        for w, q in adjs_weighted[p]:
            if not visited[q] and (pq.vals[q] is None or pq.vals[q] > w):
                edge_to[q] = p
                weights[q] = w
                pq.put(q, w)
    pq = IndexMinPQ(n)
    visited = [False] * n
    edge_to = [None] * n
    weights = [None] * n
    pq.put(0, 0.0)
    while pq.n > 0:
        i, _ = pq.pop()
        if not visited[i]:
            _visit(i)
    return sorted([(min(i, edge_to[i]), max(i, edge_to[i]), weights[i]) for i in range(1, n)])


def minSpanningTreeKruskul(edges, n):
    import heapq
    pq = [e for e in edges]
    heapq.heapify(pq)
    uf = UF(n)
    tree = []
    while pq and len(tree) < n - 1:
        w, i, j = heapq.heappop(pq)
        if not uf.connected(i, j):
            tree.append((min(i, j), max(i, j), w))
            uf.union(i, j)
    return sorted(tree)


# graph
def get_adjs_from_adjlist(adj_list, n):
    adjs = {v: [] for v in range(n)}
    for t in adj_list:
        adjs[t[0]].append(t[1])
    return adjs


def topological_sort(adjs, n):
    def _dfs(v):
        visited[v] = True
        for w in adjs[v]:
            if not visited[w]:
                _dfs(w)
        p_order.append(v)
    visited = [False] * n
    p_order = []
    for s in adjs:
        if not visited[s]:
            _dfs(s)
    return p_order[::-1]


def get_cycles(adjs, n):
    def _dfs(v):
        visited[v] = True
        on_stack[v] = True
        for w in adjs[v]:
            if on_stack[w]:
                cycle = [w]
                v_ = v
                while v_ != w:
                    cycle.insert(0, v_)
                    v_ = edge_to[v_]
                cycle.insert(0, w)
                cycles.append(cycle)
                continue
            if not visited[w]:
                edge_to[w] = v
                _dfs(w)
        on_stack[v] = False
    visited = [False] * n
    on_stack = [False] * n
    edge_to = [None] * n
    cycles = []
    for s in adjs:
        if not visited[s]:
            _dfs(s)
    return cycles


def get_all_cycles(adjs, n):
    def _dfs(v):
        visited[v] = True
        on_stack[v] = True
        for w in adjs[v]:
            if on_stack[w]:
                cycle = [w]
                v_ = v
                while v_ != w:
                    cycle.insert(0, v_)
                    v_ = edge_to[v_]
                cycle.insert(0, w)
                cycles.append(cycle)
                continue
            edge_to[w] = v
            _dfs(w)
        on_stack[v] = False
    visited = [False] * n
    on_stack = [False] * n
    edge_to = [None] * n
    cycles = []
    for s in adjs:
        if not visited[s]:
            _dfs(s)
    return cycles


def strongly_connected_compoents(adjs, n):
    # Kosaraju-Sharir algorithm
    def _dfs(v):
        ids[v] = cnt
        for w in adjs[v]:
            if ids[w] is None:
                _dfs(w)
    def rev_graph(adjs):
        adjs_rev = {s: [] for s in range(n)}
        for s in adjs:
            for w in adjs[s]:
                adjs_rev[w].append(s)
        return adjs_rev
    rp_order = topological_sort(adjs, n)
    adjs = rev_graph(adjs)
    ids = [None] * n
    cnt = 0
    for s in rp_order:
        if ids[s] is None:
            _dfs(s)
            cnt += 1
    return ids


def findOrder(strings, k):
    """ 
    N = 5, K = 4
    dict = {"baa","abcd","abca","cab","cad"}
    Output:
    1
    Explanation:
    Here order of characters is
    'b', 'd', 'a', 'c' Note that words are sorted
    and in the given language "baa" comes before
    "abcd", therefore 'b' is before 'a' in output.
    Similarly we can find other orders.
    """
    from collections import defaultdict
    adjs = defaultdict(set)
    for i in range(k+1):
        for s1, s2 in zip(strings[:-1], strings[1:]):
            if s1[:i] == s2[:i] and s1[i] != s2[i] and s1[i] not in adjs[s2[i]]:
                adjs[s1[i]].add(s2[i])
    def _dfs(v):
        visited[v] = True
        for w in adjs[v]:
            if not visited[w]:
                _dfs(w)
        p_order.append(v)

    visited = {k: False for k in adjs}
    p_order = []
    for v in adjs:
        if not visited[v]:
            _dfs(v)
    return p_order[::-1]


def findSignatureCounts(arr):
    # Write your code here
    ids = [i for i in range(len(arr))]
    sizes = [1 for _ in range(len(arr))]
    def _find(i):
        while i != ids[i]:
            i_ = i
            i = ids[i]
            ids[i_] = ids[i]
        return i
    def _connected(i, j):
        return ids[i] == ids[j]
    def _union(i, j):
        root_i = _find(i)
        root_j = _find(j)
        if root_i == root_j:
            return
        if sizes[i] < sizes[j]:
            ids[root_i] = root_j
            sizes[root_j] += sizes[root_i]
        else:
            ids[root_j] = root_i
            sizes[root_i] += sizes[root_j]
    for i, j in enumerate(arr):
        if not _connected(i, j-1):
            _union(i, j-1)
    return [sizes[_find(i)] for i in range(len(arr))]


def findSignatureCountsDFS(arr):
    # Write your code here
    def _dfs(i, cnt):
        visited[i] = True
        j = arr[i] - 1
        if not visited[j]:
            cnt = _dfs(j, cnt+1)
        cnts[i] = cnt
        return cnt
    n = len(arr)
    visited = [False] * n
    cnts = [None] * n
    for a in arr:
        i = a - 1
        if visited[i]:
            continue
        _dfs(i, 1)
    return cnts


def shortestPath(adjs, s, n):
    def _visit(p):
        visited[p] = True
        for w, q in adjs[p]:
            if not visited[q]:
                dist = dist_to[p] + w
                if pq.vals[q] is None or dist < pq.vals[q]:
                    pq.put(q, dist)
                    edge_to[q] = p
                    dist_to[q] = dist
    pq = IndexMinPQ(n)
    visited = [False] * n
    edge_to = [None] * n
    dist_to = [None] * n
    dist_to[s] = 0.0
    pq.put(s, 0.0)
    while pq.n > 0:
        i, _ = pq.pop()
        if not visited[i]:
            _visit(i)
    return sorted([(i, edge_to[i], dist_to[i]) for i in range(n)])


def treeOfCoprimes(nums, edges):
    """
    https://leetcode.com/problems/tree-of-coprimes/
    There is a tree (i.e., a connected, undirected graph that has no cycles) consisting of n nodes numbered from 0
    to n - 1 and exactly n - 1 edges. Each node has a value associated with it, and the root of the tree is node 0.
    To represent this tree, you are given an integer array nums and a 2D array edges. Each nums[i] represents the ith
    node's value, and each edges[j] = [uj, vj] represents an edge between nodes uj and vj in the tree.
    Two values x and y are coprime if gcd(x, y) == 1 where gcd(x, y) is the greatest common divisor of x and y.
    An ancestor of a node i is any other node on the shortest path from node i to the root. A node is not considered an
    ancestor of itself.
    Return an array ans of size n, where ans[i] is the closest ancestor to node i such that nums[i] and nums[ans[i]] are
    coprime, or -1 if there is no such ancestor.

    :return:
    """

    def _gcd(a, b):
        return a if b == 0 else _gcd(b, a % b)

    def _dfs_2(i, path, d):
        for j in adjs[i]:
            adjs[j].remove(i)
            path_ = path.copy()
            path_[nums[i]] = (i, d)
            _dfs_2(j, path_, d + 1)

        max_depth = -1
        parent = -1
        for num in valids[nums[i]]:
            if num in path and path[num][1] > max_depth:
                max_depth = path[num][1]
                parent = path[num][0]
        parents[i] = parent

    n = len(nums)
    adjs = {i: [] for i in range(n)}
    for n1, n2 in edges:
        adjs[n1].append(n2)
        adjs[n2].append(n1)

    uniq_nums = set(nums)
    valids = {num: set() for num in uniq_nums}
    for n1 in uniq_nums:
        for n2 in uniq_nums:
            if _gcd(n1, n2) == 1:
                valids[n1].add(n2)
                valids[n2].add(n1)

    parents = [-1] * n
    _ = _dfs_2(0, {}, 0)

    return parents


def findRedundantDirectedConnection(edges: List[List[int]]) -> List[int]:
    # https://leetcode.com/problems/redundant-connection-ii/
    #         return extra_edge
    pass

# https://leetcode.com/problems/find-eventual-safe-states/submissions/1349086285/

# binary trees
class SegmentTreeArrayImplementation:
    def __init__(self, arr):
        self.tree = None
        self.n = 0
        self._build_tree(arr)

    def _build_tree(self, arr):
        n = len(arr)
        self.n = n
        self.tree = [0] * 2 * n
        for i in range(n):
            self.tree[i + n] = arr[i]
        for i in range(n-1, 0, -1):
            self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]

    def update(self, i, v):
        i += self.n
        self.tree[i] = v
        while i > 1:
            self.tree[i >> 1] = self.tree[i] + self.tree[i ^ 1]
            i >>= 1

    def query(self, l, r):
        l += self.n
        r += self.n
        sum_ = 0
        while l < r:
            if r & 1:
                r -= 1
                sum_ += self.tree[r]
            if l & 1:
                sum_ += self.tree[l]
                l += 1
            l >>= 1
            r >>= 1
        return sum_


class SegmentTree2DArrayImplementation:
    def __init__(self, arr_2d):
        self.tree = None
        self.n, self.m = 0, 0
        self._build(arr_2d)

    def _build(self, arr2d):
        self.m = len(arr2d)
        self.n = len(arr2d[0])
        self.tree = [[0 for _ in range(2 * self.n)] for _ in range(2 * self.m)]
        # building segment tree in one dimension
        for i in range(self.m):
            for j in range(self.n):
                self.tree[i+self.m][j+self.n] = arr2d[i][j]
            for j in range(self.n-1, 0, -1):
                self.tree[i+self.m][j] = self.tree[i+self.m][j << 1] + self.tree[i+self.m][j << 1 | 1]
        # building segment tree in other dimension by merging arrays
        for i in range(self.m-1, 0, -1):
            for j in range(self.n*2):
                self.tree[i][j] = self.tree[i << 1][j] + self.tree[i << 1 | 1][j]

    def put(self, i, j, v):
        i += self.m
        j += self.n
        self.tree[i][j] = v
        while j > 1:
            self.tree[i][j >> 1] = self.tree[i][j] + self.tree[i][j ^ 1]
            j >>= 1
        while i > 1:
            for j_ in range(self.n + j):
                self.tree[i >> 1][j_] = self.tree[i][j_] + self.tree[i ^ 1][j_]
            i >>= 1

    def get(self, l, r, t, b):
        t += self.m
        b += self.m
        res = 0
        while t < b:
            if t & 1:
                l_ = l + self.n
                r_ = r + self.n
                while l_ < r_:
                    if r_ & 1:
                        r_ -= 1
                        res += self.tree[t][r_]
                    if l_ & 1:
                        res += self.tree[t][l_]
                        l_ += 1
                    l_ >>= 1
                    r_ >>= 1
                t += 1

            if b & 1:
                b -= 1
                l_ = l + self.n
                r_ = r + self.n
                while l_ < r_:
                    if r_ & 1:
                        r_ -= 1
                        res += self.tree[b][r_]
                    if l_ & 1:
                        res += self.tree[b][l_]
                        l_ += 1
                    l_ >>= 1
                    r_ >>= 1
            t >>= 1
            b >>= 1
        return res


class SegmentTreeNode:
    def __init__(self, interval, v):
        self.interval = interval
        self.val = v
        self.left = None
        self.right = None


class SegmentTree1D:
    def __init__(self):
        self.root = None

    def put(self, interval, v):
        self.root = self._put(self.root, interval, v)

    def _put(self, root, interval, v):
        if root is None:
            return SegmentTreeNode(interval, v)
        tl, tc, tr = self._split((root.interval, root.val), (interval, v))
        root.interval, root.val = tc
        if tl:
            root.left = self._put(root.left, tl[0], tl[1])
        if tr:
            root.right = self._put(root.right, tr[0], tr[1])
        return root

    def get(self, i):
        return self._get(self.root, i)

    def _get(self, root, i):
        if not root:
            return None
        l, r = root.interval
        if l <= i < r:
            return root.val
        elif i < l:
            return self._get(root.left, i)
        else:
            return self._get(root.right, i)

    def _split(self, tv_root, tv):
        # tvx : ((l, r), value)
        t1, v1 = tv_root
        t2, v2 = tv
        if t1[1] <= t2[0]:
            return None, tv_root, tv
        if t1[0] >= t2[1]:
            return tv, tv_root, None
        tvl, tvc, tvr = None, None, None
        if t1[0] <= t2[0]:
            if t1[0] != t2[0]:
                tvl = ((t1[0], t2[0]), v1)
            i_ol = min(t1[1], t2[1])
            tvc = ((t2[0], i_ol), v2)
            if t2[1] != t1[1]:
                vr = v1 if t1[1] > t2[1] else v2
                tvr = ((i_ol, max(t1[1], t2[1])), vr)
        else:
            tvl = ((t2[0], t1[0]), v2)
            i_ol = min(t1[1], t2[1])
            tvc = ((t1[0], i_ol), v2)
            if t2[1] != t1[1]:
                vr = v1 if t1[1] > t2[1] else v2
                tvr = ((i_ol, max(t1[1], t2[1])), vr)
        return tvl, tvc, tvr


class SegmentTree2D:
    def __init__(self):
        self.root = None

    def put(self, l, r, t, b, v):
        pass

    def get(self, i, j):
        pass


class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def put(self, word):
        root = self.root
        for c in word:
            ind = ord(c) - ord('a')
            if not root.children[ind]:
                root.children[ind] = TrieNode()
            root = root.children[ind]
        root.is_end = True

    def search(self, word):
        root = self.root
        for c in word:
            ind = ord(c) - ord('a')
            if not root.children[ind]:
                return False
            root = root.children[ind]
        return root.is_end


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def preOrderIter(root):
    stk = []
    pre_order = []
    while root or stk:
        if root:
            pre_order.append(root.data)
            stk.append(root)
            root = root.left
        else:
            root = stk.pop()
            root = root.right
    return pre_order


def inOrderIter(root):
    stk = []
    in_order = []
    while root or stk:
        if root:
            stk.append(root)
            root = root.left
        else:
            root = stk.pop()
            in_order.append(root.data)
            root = root.right
    return in_order


def postOrderIter(root):
    stk = []
    post_order = []
    while root or stk:
        if root:
            if root.right:
                stk.append(root.right)
            stk.append(root)
            root = root.left
        else:
            root = stk.pop()
            if stk and root.right and stk[-1] == root.right:
                stk.pop()
                stk.append(root)
                root = root.right
            else:
                post_order.append(root.data)
                root = None
    return post_order


def level_order(tree):
    next_level = [tree]
    levels = []
    while next_level:
        levels.append([n.val for n in next_level])
        cur_level, next_level = next_level, []
        while cur_level:
            node = cur_level.pop(0)
            if node.l:
                next_level.append(node.l)
            if node.r:
                next_level.append(node.r)
    return levels


def tree_from_inorder_preorder(inorder, preorder):
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    ind = inorder.index(preorder[0])
    root.l = tree_from_inorder_preorder(inorder[:ind], preorder[1:ind+1])
    root.r = tree_from_inorder_preorder(inorder[ind+1:], preorder[ind+1:])
    return root


def tree_from_inorder_postorder(inorder, postorder):
    if not inorder:
        return None
    root = TreeNode(postorder[-1])
    ind = inorder.index(postorder[-1])
    root.l = tree_from_inorder_postorder(inorder[:ind], postorder[:ind])
    root.r = tree_from_inorder_postorder(inorder[ind+1:], postorder[ind:-1])
    return root


def tree_from_preorder_postorder(preorder, postorder):
    if not preorder:
        return None
    root = TreeNode(postorder[-1])
    if len(preorder) == 1:
        return root
    ind = preorder.index(postorder[-2])
    root.l = tree_from_preorder_postorder(preorder[1:ind], postorder[:ind-1])
    root.r = tree_from_preorder_postorder(preorder[ind:], postorder[ind-1:len(postorder)-1])
    return root


class TreeCodec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return "*"
        return f"{root.data}|{self.serialize(root.left)}|{self.serialize(root.right)}"

    def deserialize(self, tree_string):
        def _deserialize(val):
            if val is None:
                return None
            root = TreeNode(val)
            root.left = _deserialize(pre_order.pop(0))
            root.right = _deserialize(pre_order.pop(0))
            return root
        pre_order = [int(c) if c.isdigit() else None for c in tree_string.split("|")]
        return _deserialize(pre_order.pop(0))


# https://leetcode.com/problems/recover-binary-search-tree/submissions/1348193114/


def dupSubTree(tree):
    # find if there is duplicate subtree larger than one
    def _search(root):
        nonlocal found
        if found:
            return ""
        if root is None:
            return "*"
        s = f"{root.data}|{_search(root.left)}|{_search(root.right)}"
        valid_s = [c for c in s.split("|") if c != "*"]
        if s in mem:
            found = True
            return ""
        if len(valid_s) > 1:
            mem.add(s)
        return s

    mem = set()
    found = False
    _ = _search(tree)

    return found


def mergeBSTreesIterative(tree1, tree2):
    stk1, stk2 = [], []
    root1, root2 = tree1, tree2
    in_order = []
    while root1 or stk1 or root2 or stk2:
        if root1 or root2:
            if root1:
                stk1.append(root1)
                root1 = root1.left
            if root2:
                stk2.append(root2)
                root2 = root2.left
        else:
            if stk1 and stk2:
                if stk1[-1].data < stk2[-1].data:
                    root1 = stk1.pop()
                    in_order.append(root1.data)
                    root1 = root1.right
                else:
                    root2 = stk2.pop()
                    in_order.append(root2.data)
                    root2 = root2.right
            elif stk1:
                root1 = stk1.pop()
                in_order.append(root1.data)
                root1 = root1.right
            else:
                root2 = stk2.pop()
                in_order.append(root2.data)
                root2 = root2.right
    return in_order


class IntervalNode:
    def __init__(self, interval, data=None):
        self.data = data
        self.interval = interval
        self.right = None
        self.left = None
        self.max = interval[1]


class IntervalTree:
    def __init__(self):
        self.root = None

    def put(self, interval, data=None):
        def _put(root):
            if root is None:
                return IntervalNode(interval, data)
            if interval[0] < root.interval[0]:
                root.left = _put(root.left)
            else:
                root.right = _put(root.right)
            root.max = max(root.max, interval[1])
            return root
        self.root = _put(self.root)

    def get(self, interval, black_list=None):
        def _get(root):
            if root is None:
                return
            if self._match(interval, root.interval):
                return root.data, root.interval
            elif interval[0] < root.interval[0] and root.left and root.left.max >= interval[1]:
                return _get(root.left)
            elif interval[0] >= root.interval[0] and root.right and root.right.max >= interval[1]:
                return _get(root.right)
            return None

        return _get(self.root)

    @staticmethod
    def _match(interval1, interval2):
        return interval2[0] <= interval1[0] and interval1[1] <= interval2[1]


def distanceK(root, target, k):
    # find nodes in distance k from the target
    def _dfs(node, dist_up):
        if node is None:
            return -1

        if node.data == target:
            _dfs(node.left, 1)
            _dfs(node.right, 1)
            return 1

        if dist_up is None:

            ans = _dfs(node.left, None)
            if ans == k:
                dist_k_nodes.append(node)
            if 0 < ans < k:
                _ = _dfs(node.right, ans + 1)
            
            if ans > 0:
                return ans + 1
            
            ans = _dfs(node.right, None)
            if ans == k:
                dist_k_nodes.append(node)
            if 0 < ans < k:
                _ = _dfs(node.left, ans + 1)

            return ans + 1 if ans > 0 else -1

        else:

            if dist_up == k:
                dist_k_nodes.append(node)
            elif dist_up < k:
                _dfs(node.left, dist_up + 1)
                _dfs(node.right, dist_up + 1)
            return -1
    
    dist_k_nodes = []
    _dfs(root, None)
    return dist_k_nodes


# dp
def textJust_bu(words, width):
    n = len(words)
    dp = [0] + [float('inf')] * n
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            width_split = sum([len(w) for w in words[j - 1:i]]) + i - j
            if width_split > width:
                break
            cost = (width - width_split) ** 2
            dp[i] = min(dp[i], cost + dp[j - 1])
    return dp[-1]


def textjust_td(words, w):
     def _dfs(i, l):
         if i == n:
             return 0 if l == 0 else (w - l)**2
         l += len(words[i]) + (l > 0)
         if (i, l) in mem:
             return mem[(i, l)]
         if l > w:
             cost = float('inf')
         else:
             cost = min((w - l)**2 + _dfs(i+1, 0), _dfs(i+1, l))
         mem[(i, l)] = cost
         return cost
     n = len(words)
     mem = {}
     return _dfs(0, 0)


def stockBuySellTopK2(prices, k):
        
    dp = [0] * len(prices)
    for _ in range(k):
        dp_ = [d for d in dp]
        for i in range(1, len(prices)):
            dp[i] = max(dp_[i], dp[i - 1])
            for j in range(i):
                if prices[i] > prices[j]:
                    dp[i] = max(dp[i], dp_[j] + prices[i] - prices[j])
    return dp[-1]


def stockBuySellOptim(prices, k):
    def _get_extrms(arr, n):
        extrms = []

        i = 0
        while i < n-1:
            while i < n - 1 and arr[i+1] <= arr[i]:
                 i += 1            
            if i == n-1:
                break
            extrms.append(arr[i])            
            while i < n - 1 and arr[i+1] >= arr[i]:
                i += 1
            extrms.append(arr[i])

        return extrms
    
    extrms = _get_extrms(prices, len(prices))
    if not extrms:
        return 0    
        
    return stockBuySellTopK2(extrms, k)


def stockBuySellOptimDFs(prices, k):
    from functools import cache

    def get_extrms(prices):
        n = len(prices)
        if n <= 1:
            return prices
        extrms = []
        i = 0
        while True:
            while i < n-1 and prices[i+1] <= prices[i]:
                i += 1
            if i == n - 1:
                break
            extrms.append(prices[i])
            while i < n-1 and prices[i+1] >= prices[i]:
                i += 1
            extrms.append(prices[i])
        return extrms

    @cache
    def _dfs(i, ind, rem):
        if ind == n or rem == 0 or i > ind:
            return 0

        profit = max(_dfs(i, ind+1, rem), _dfs(i+1, ind, rem))
        if prices[ind] > prices[i]:
            profit = max(profit, prices[ind] - prices[i] + _dfs(ind, ind, rem-1))

        return profit

    prices = get_extrms(prices)
    n = len(prices)
    return _dfs(0, 0, k)


def maxPickFromEndsTwoPlayer(arr):
    m = len(arr)
    dp = [[0 for _ in range(m)] for _ in range(m)]
    for l in range(m):
        for i in range(m - l):
            if l < 2:
                dp[i][i+l] = max(arr[i], arr[i+l])
                continue
            dp[i][i+l] = max(min(dp[i+1][i+l-1], dp[i+2][i+l]) + arr[i], 
                             min(dp[i][i+l-2], dp[i+1][i+l-1]) + arr[i+l])
    return dp[0][-1]


def maxPickFromEndsTwoPlayerDfs(arr):
    @cache
    def _dfs(l, r):
        if r - l <= 1:
            return max(arr[l:r+1])
        return max(arr[l] + min(_dfs(l+1, r-1), _dfs(l+2, r)), arr[r] + min(_dfs(l+1, r-1), _dfs(l, r-2)))

    return _dfs(0, len(arr) - 1)


def palindormPartitioningON3(s):
    m = len(s)
    dp = [[0 if i >= j else float('inf') for j in range(m)] for i in range(m)]
    for l in range(1, m):
        for i in range(m - l):
            if s[i] == s[i + l] and dp[i + 1][i + l - 1] == 0:
                dp[i][i + l] = 0
                continue
            for j in range(i, i + l):
                dp[i][i + l] = min(dp[i][i + l], dp[i][j] + dp[j + 1][i + l] + 1)
    return dp


def palindromPartitioningON2(s):
    dp = [-1] + len(s) * [len(s)]
    splits = [[]] + [[]] * len(s)
    for i in range(1, len(dp)):
        k = i
        j = 0
        while i + j < len(dp) and k - j > 0 and s[k - j - 1] == s[i + j - 1]:
            if  dp[k - j - 1] + 1 < dp[i+j]:
                splits[i+j] = splits[k-j-1] + [k-j]
            dp[i + j] = min(dp[i + j], dp[k - j - 1] + 1)
            j += 1
        k = i - 1
        j = 0
        while i + j < len(dp) and k - j > 0 and s[k - j - 1] == s[i + j - 1]:
            if dp[k - j - 1] + 1 < dp[i + j]:
                splits[i+j] = splits[k - j - 1] + [k - j]
            dp[i + j] = min(dp[i + j], dp[k - j - 1] + 1)
            j += 1
    return dp[-1], splits


def burstingBalloon(balloons):
    n = len(balloons)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for m in range(n):
        for i in range(n - m):
            l = 1 if i == 0 else balloons[i - 1]
            r = 1 if i + m == n - 1 else balloons[i + m + 1]
            for j in range(i, i + m + 1):
                dp[i][i + m] = max(dp[i][i + m],
                                   (dp[i][j - 1] if j > i else 0) +
                                   l * balloons[j] * r +
                                   (dp[j + 1][i + m] if j < i + m else 0))
    return dp


def maxSubsqureSidesFilled(mtrx):
    # xos = ["O O O O X".split(), "X O X X X".split(), "X O X O X".split(), "X X X X X".split(), "O O X X X".split()]
    m, n = len(mtrx), len(mtrx[0])
    dp = [[(0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if mtrx[i - 1][j - 1] == "O":
                continue
            max_left = dp[i][j - 1][0] + 1
            max_top = dp[i - 1][j][1] + 1
            min_dim = 1
            for k in range(1, min(max_top, max_left)):
                if dp[i - k][j][0] >= k + 1 and dp[i][j - k][1] >= k + 1:
                    min_dim = k + 1
            dp[i][j] = (max_left, max_top, min_dim)
    return dp


def regexMatch(s, p):
    n, m = len(p), len(s)
    dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(0, m + 1):
        for j in range(1, n + 1):
            if i == 0:
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j-2]
                continue
            if s[i - 1] == p[j - 1] or p[j - 1] == ".":
                dp[i][j] = dp[i - 1][j - 1]
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 2] | (dp[i - 1][j] and (p[j - 2] == s[i - 1] or p[j - 2] == "."))
    return dp[-1][-1]


def wildCardMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False for _ in range(n+1)] for _ in range(m+1)]
    dp[0][0] = True
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == p[j-1] or p[j-1] == "?":
                dp[i][j] = dp[i-1][j-1]
            if p[j-1] == "*":
                dp[i][j] = dp[i-1][j] | dp[i][j-1]
    return dp


def eggDrop(k, n):
    # code here
    dp = [i for i in range(n + 1)]
    for k_ in range(2, k + 1):
        dp_prev = [d for d in dp]
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                dp[i] = min(dp[i], 1 + max(dp_prev[j - 1], dp[i - j]))
    return dp[-1]


def superEggDrop(K, N):
    def _linSearch(m, j):
        while m < j and max(dp_prev[m - 1], dp[j - m]) > max(dp_prev[m], dp[j - m - 1]):
            m += 1
        return m
    dp = [j for j in range(N + 1)]
    for i in range(2, K + 1):
        dp_prev = [d for d in dp]
        m_optim = 1
        for j in range(1, N + 1):
            m_optim = _linSearch(m_optim, j)
            # print(i, j, m_optim)
            dp[j] = max(dp_prev[m_optim - 1], dp[j - m_optim]) + 1
    return dp[-1]


def sequenceCount(s1, s2, bu=True):
    # S = "geeksforgeeks" , T = "ge"
    # [ge], [ ge], [g e], [g e] [g e] and [ g e].
    def _td(i, j):
        if j == m and i <= n:
            return 1
        if n - i < m - j:
            return 0
        if (i, j) in mem:
            return mem[(i, j)]
        cnts = _td(i+1, j)
        if s1[i] == s2[j]:
            cnts += _td(i+1, j+1)
        mem[(i, j)] = cnts
        return cnts

    def _bu():
        dp = [0 for _ in range(m + 1)]
        dp[0] = 1
        for i in range(1, n + 1):
            dp_prev = [d for d in dp]
            for j in range(1, m + 1):
                dp[j] = dp_prev[j] + (s1[i - 1] == s2[j - 1]) * dp_prev[j - 1]
        return dp[-1]

    mem = {}
    n, m = len(s1), len(s2)

    return (_bu() if bu else _td(0, 0)) % (10 ** 9 + 7)


def findMinInsertionsPalindrom(S, td=False):
    def _budp(s):
        n = len(s)
        dp = [[0 if i >= j else n for j in range(n)] for i in range(n)]
        for l in range(1, n):
            for i in range(n - l):
                if s[i] == s[i+l]:
                    dp[i][i+l] =  dp[i+1][i+l-1]
                else:
                    dp[i][i+l] = min(dp[i+1][i+l], dp[i][i+l-1]) + 1

        return dp[0][-1]

    def _tddp(s):
        @cache
        def _dfs(i, j):
            # stop condition
            if i == j or (j - i == 1 and s[i] == s[j]):
                return 0

            # formula
            if s[i] == s[j]:
                cnt = _dfs(i+1, j-1)
            else:
                cnt = min(_dfs(i+1, j), _dfs(i, j-1)) + 1

            return cnt

        return _dfs(0, len(s) - 1)
    
    return _tddp(S) if td else _budp(S)


def minTime(arr, n, k, use_bs=True):
    """
    n = 5
    k = 3
    arr[] = {5,10,30,20,15}
    Output: 35
    Explanation: The most optimal way will be:
    Painter 1 allocation : {5,10}
    Painter 2 allocation : {30}
    Painter 3 allocation : {20,15}
    Job will be done when all painters finish
    i.e. at time = max(5+10, 30, 20+15) = 35
    """
    def _budp():
        print("n, k: ", n, k)
        dp = [0] + [float('inf')] * n
        for _ in range(k):
            dp_prev = [d for d in dp]
            for i in range(1, n + 1):
                for j in range(i, 0, -1):
                    dp[i] = min(dp[i], max(dp_prev[j - 1], sum(arr[j - 1:i])))
        return dp[-1]

    def _binarySearch(arr, n, k):
        arr = [int(a) for a in arr]
        def _num_painters(max_l):
            n_painters = 1
            total = 0
            for a in arr:
                total += a
                if total > max_l:
                    total = a
                    n_painters += 1
            return n_painters

        l, r = max(arr), sum(arr)
        while l < r:
            m = (l + r) // 2
            n_painters = _num_painters(m)
            if n_painters > k:
                l = m + 1
            else:
                r = m
        segments = []
        total = 0
        cur_segment = []
        for a in arr:
            if total + a > r:
                segments.append(cur_segment)
                total = a
                cur_segment = [a]
                continue
            total += a
            cur_segment.append(a)

        if cur_segment:
            segments.append(cur_segment)

        return max([sum(segment) for segment in segments])

    return _binarySearch(arr, n, k) if use_bs else _budp()


def findMaximumNum(s, k):
    """
    takes a string s and an integer k, and attempts to return the maximum possible number that can be 
    formed by swapping the characters in the string s at most k times. 

    Example Usage
    If you use this function with inputs like s = "1234" and k = 2, it will explore ways to make up to 
    2 swaps in the string "1234" to form the largest possible number. 
    """
    @cache
    def _dfs(cur_s, rem):
        if rem == 0 or len(cur_s) == 0:
            return cur_s

        pick_choice = max(list(cur_s))
        if cur_s[0] == pick_choice:
            return cur_s[0] + _dfs(cur_s[1:], rem)

        max_num = cur_s
        for i in range(len(cur_s)):
            if cur_s[i] == pick_choice:
                max_num = max(max_num, cur_s[i] + _dfs(cur_s[1:i] + cur_s[0] + cur_s[i+1:], rem - 1))

        return max_num

    return _dfs(s, k)


def knapsack01_bu(n, w, wt, val):
    """
    problem: 0/1 knapsack
    Given weights and values of n items, put these items in a knapsack of capacity w 
    to get the maximum total value in the knapsack.

    n = 3, w = 4, wt = [4, 5, 1], val = [1, 2, 3]

    Output: 3

    description:
    The maximum value that can be achieved is 3 with a weight of 4, by taking the items with weight 1 and 3.
    """
    dp = [[0 for _ in range(w + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, w + 1):
            dp[i][j] = dp[i - 1][j]
            if wt[i - 1] <= j:
                dp[i][j] = max(dp[i][j], val[i - 1] + dp[i - 1][j - wt[i - 1]])
                
    return dp[-1][-1]


def knapsack01_td(n, w, wt, val):
    @cache
    def _dfs(i, rem):
        if i == n or rem == 0:
            return 0

        if wt[i] > rem:
            return _dfs(i+1, rem)

        return max(_dfs(i+1, rem), val[i] + _dfs(i+1, rem-wt[i]))

    return _dfs(0, w)


def profitableSchemes(n, minProfit, group, profit):
    """
    description:
    There are G people in a gang, and a list of various crimes they could commit. 
    The i-th crime generates a profit[i] and requires group[i] gang members to participate. 
    If a gang member participates in one crime, that member can't participate in another crime. 
    Find out how many crimes can be committed such that the total profit is at least minProfit. 
    Each crime can be completed only once.
    
    n: total number of gang members available
    minProfit: minimum profit to be made
    group: number of gang members required for each crime
    profit: profit for each crime
    
    Example:
    n = 5, minProfit = 3, group = [2,2], profit = [2,3]
    Output: 2
    Explanation: To make a profit of at least 3, the group could either commit crimes 0 and 1, or just crime 1.
    In total, there are 2 schemes.
    """
    def _td(n, minProfit, group, profit):
        # TLE
        @cache
        def _dfs(i, cur_profit, cur_grp):
            if i == m:
                return 1 if cur_grp <= n and cur_profit >= minProfit else 0

            cnt = _dfs(i+1, cur_profit, cur_grp)
            if group[i] + cur_grp <= n:
                cnt += _dfs(i+1, cur_profit + profit[i], cur_grp + group[i])

            return cnt

        m = len(group)

        return _dfs(0, 0, 0)

    def _bu(n, minProfit, group, profit):
        dp = [[0 for j in range(n + 1)] for i in range(minProfit + 1)]
        dp[0][0] = 1
        for g, p in zip(group, profit):
            dp_prev = [t.copy() for t in dp]
            for i in range(minProfit + 1):                
                for j in range(g, n + 1):
                    dp[i][j] += dp_prev[max(i - p, 0)][j - g]

        return sum(dp[-1]) % (10 ** 9 + 7)

    def _bu2(n, minProfit, group, profit):
        dp = [[0 for j in range(n + 1)] for i in range(minProfit + 1)]
        dp[0][0] = 1
        for g, p in zip(group, profit):
            # we can go backwards to avoid overwriting and use only one dp array (thanks GPT4o!)
            for i in range(minProfit, -1, -1):                
                for j in range(n-g, -1, -1):
                    dp[min(i+p, minProfit)][j+g] += dp[i][j]
                    
        return sum(dp[-1]) % (10 ** 9 + 7)
    
    return _bu2(n, minProfit, group, profit)


def jobScheduling(startTime, endTime, profit):
    n = len(startTime)
    start_ends = sorted(list(zip(startTime, endTime, profit)), key=lambda t: t[1])
    dp = [0] * (n + 1)
    for i in range(1, n+1):
        j = i-1
        while j > 0 and start_ends[j-1][1] > start_ends[i-1][0]:
            j -= 1
        dp[i] = max(dp[i-1], dp[j] + start_ends[i-1][2])
    return dp[-1]


def calculateMinimumHP(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    lives = [[float('inf') for _ in range(n+1)] for _ in range(m+1)]
    lives[m-1][n] = 1
    lives[m][n-1] = 1
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            lives[i][j] = max(1, min(lives[i+1][j], lives[i][j+1]) - dungeon[i][j])
    print(lives)
    return lives[0][0]


def catAndMouse(graph):
    
    def catMouseGame_dfs_td(graph: List[List[int]]) -> int:
        # not working for new test cases on leetcode possibly with cycle!
        # dfs cannot solve this problem for all edge cases
        @cache
        def helper(turn, cat, mouse):
            if turn == 0:
                return 0

            options = set()

            # Cat's turn
            if turn & 1:
                if mouse in graph[cat] or cat == mouse:
                    return 2
                for c in graph[cat] - {0}:
                    options.add(helper(turn - 1, c, mouse))
                    if 2 in options:
                        return 2
                return min(options)

            # Mouse's turn
            if 0 in graph[mouse]:
                return 1
            for m in graph[mouse]:
                options.add(helper(turn - 1, cat, m))
                if 1 in options:
                    return 1

            return min(options)

        n = len(graph) + (len(graph) & 1)  # all possible cases: mouse and cat try all edges
        graph = {i: set(g) for i, g in enumerate(graph)}

        return helper(n, 2, 1)

    def catMouseGame_percolate_bu(graph):
        import collections

        N = len(graph)

        # What nodes could play their turn to
        # arrive at node (m, c, t) ?
        def parents(m, c, t):
            if t == 2:
                for m2 in graph[m]:
                    yield m2, c, 3-t
            else:
                for c2 in graph[c]:
                    if c2 > 0:
                        yield m, c2, 3-t

        DRAW, MOUSE, CAT = 0, 1, 2
        color = collections.defaultdict(int)

        # degree[node] : the number of neutral children of this node
        degree = {}
        for m in range(N):
            for c in range(N):
                degree[m,c,1] = len(graph[m])
                degree[m,c,2] = len(graph[c]) - (0 in graph[c])

        # enqueued : all nodes that are colored
        queue = collections.deque([])
        for i in range(N):
            for t in range(1, 3):
                color[0, i, t] = MOUSE
                queue.append((0, i, t, MOUSE))
                if i > 0:
                    color[i, i, t] = CAT
                    queue.append((i, i, t, CAT))

        # percolate
        while queue:
            # for nodes that are colored :
            i, j, t, c = queue.popleft()
            # for every parent of this node i, j, t :
            for i2, j2, t2 in parents(i, j, t):
                # if this parent is not colored :
                if color[i2, j2, t2] is DRAW:
                    # if the parent can make a winning move (ie. mouse to MOUSE), do so
                    if t2 == c: # winning move
                        color[i2, j2, t2] = c
                        queue.append((i2, j2, t2, c))
                    # else, this parent has degree[parent]--, and enqueue if all children
                    # of this parent are colored as losing moves
                    else:
                        degree[i2, j2, t2] -= 1
                        if degree[i2, j2, t2] == 0:
                            color[i2, j2, t2] = 3 - t2
                            queue.append((i2, j2, t2, 3 - t2))

        return color[1, 2, 1]

    catMouseGame_percolate_bu(graph)


def numPermsDISequence(self, s: str) -> int:
    # https://leetcode.com/problems/valid-permutations-for-di-sequence/
    @cache
    def fn(i, x):
        """Return number of valid permutation given x numbers smaller than previous one."""
        if i == len(s): return 1
        if s[i] == "D":
            if x == 0: return 0  # cannot decrease
            return fn(i, x - 1) + fn(i + 1, x - 1)
        else:
            if x == len(s) - i: return 0  # cannot increase
            return fn(i, x + 1) + fn(i + 1, x)

    return sum(fn(0, x) for x in range(len(s) + 1)) % 1_000_000_007


def whoIsElected(n, k):
    """
    A group of students are sitting in a circle. The teacher is electing a new class president.
    The teacher does this by singing a song while walking around the circle. After the song is
    finished the student at which the teacher stopped is removed from the circle.

    Starting at the student next to the one that was just removed, the teacher resumes singing and walking around the circle.
    After the teacher is done singing, the next student is removed. The teacher repeats this until only one student is left.

    A song of length k will result in the teacher walking past k students on each round. The students are numbered 1 to n. The teacher starts at student 1.

    For example, suppose the song length is two (k=2). And there are four students to start with (1,2,3,4). The first
    student to go would be `2`, after that `4`, and after that `3`. Student `1` would be the next president in this example.

    @param n:   the number of students sitting in a circle.
    @param k:   the length (in students) of each song.
    @return:    the number of the student that is elected.
    """
    # todo: implement here
    def whoIsElected(n, k):
        # Josephus problem
        # J(n,k)=(J(n−1,k)+k)%n
        # recursive soln:
        """
        def whoIsElected(n, k):
            def josephus(n, k):
                if n == 1:
                    return 0
                else:
                    return (josephus(n-1, k) + k) % n
            
            return josephus(n, k) + 1
        """
        survivor = 0  # Start with the base case where the last remaining index is 0 for n = 1
        for i in range(2, n + 1):
            survivor = (survivor + k) % i
        return survivor + 1  # Convert from 0-based to 1-based index


"""other"""
def matchIntervieweeInterviewer(interivew_time, interviewers, interviewees):
    '''
    """
    Give time slots of interviewers and interviewees, schedule interviews.

    Input1: {Interviewer1: [[t1, t2], [t3, t4]...], interviewer2: [[t5, t6], [t7, t8]]}
    Input2: {Interviewee1: [[te1, te2], [te3, te4]...], interviewee2: [[te5, te6], [te7, te8]]}

    Interview_time = T

    Output: [[interviewer1, interviewee2, [t1, t1+T]]....]


    Interview_time = 1
    Input1: {'tom': [[1, 3], [6, 10]], 'tom2': [[2, 4], [5,6]]}
    Input2: {'in1': [[2, 3], [4, 5]], 'in2': [[1, 2], [3, 4]]}

    maxlen Output = len(interviewees)

    Output: [['tom1', 'in1', [2,3]], ['tom2', 'in2', [3, 4]]]
    """

    # find available interviwers per time slots of interviweees

    #
    '''
    pass


#https://www.hackerrank.com/challenges/the-quickest-way-up/problem
# https://leetcode.com/problems/largest-component-size-by-common-factor/
# https://leetcode.com/problems/verbal-arithmetic-puzzle/discuss/463921/python-backtracking-with-pruning-tricks
# https://leetcode.com/problems/contain-virus/
# https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
# https://leetcode.com/problems/valid-triangle-number/submissions/
# https://leetcode.com/problems/strange-printer/
# https://leetcode.com/submissions/detail/111795953/
# https://leetcode.com/problems/count-pairs-with-xor-in-a-range/discuss/1119703/Python3-trie
# https://leetcode.com/problems/find-the-duplicate-number/discuss/?currentPage=1&orderBy=most_votes&query=
# https://leetcode.com/problems/stone-game-ii/submissions/
# https://leetcode.com/problems/find-all-good-strings/discuss/655787/Python-O(nk)-KMP-with-DP-very-detailed-explanation.
# https://leetcode.com/problems/number-of-atoms/submissions/
# https://leetcode.com/problems/prison-cells-after-n-days/discuss/205684/JavaPython-Find-the-Loop-or-Mod-14
# https://leetcode.com/problems/race-car/discuss/123834/JavaC%2B%2BPython-DP-solution
# https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/submissions/
# https://leetcode.com/problems/ways-to-split-array-into-three-subarrays/submissions/
# https://leetcode.com/problems/largest-merge-of-two-strings/submissions/
# https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/
# https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/submissions/
# https://leetcode.com/problems/couples-holding-hands/
# https://leetcode.com/problems/minimum-jumps-to-reach-home/submissions/
# https://leetcode.com/problems/longest-consecutive-sequence/submissions/1329514724/
# https://leetcode.com/problems/substring-with-concatenation-of-all-words/submissions/1348026873/
# https://leetcode.com/problems/word-ladder-ii/
# https://leetcode.com/problems/cat-and-mouse-ii/
# https://leetcode.com/problems/strange-printer/solutions/233067/python-recursive-approach-with-memorization/
