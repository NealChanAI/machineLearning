"""
Python实现快速排序
"""

def quickSort(lst, left, right):
    # 数组大小小于0, 则返回None
    if len(lst) < 0: return None
    if left >= right: return None
    # 确定基准值pivot
    pivot = lst[left]
    l = left 
    r = right
    # 循环遍历
    while l < r:
        # 从右边开始往左遍历
        while l < r and lst[r] > pivot:
            r -= 1
        lst[l] = lst[r]
        # 然后从左边开始往右遍历
        while l < r and lst[l] < pivot:
            l += 1
        lst[r] = lst[l]
    lst[l] = pivot

    # 用相同的方法处理两个子序列
    quickSort(lst, left, r-1)
    quickSort(lst, r+1, right)
    return lst
