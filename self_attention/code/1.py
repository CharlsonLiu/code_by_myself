from typing import List


class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        l = len(nums)
        avg = []
        for i in range(len(nums) // 2):
            avg.append((nums[i]+nums[l-1]) / 2)
            print('min',nums[i])
        return min(avg)