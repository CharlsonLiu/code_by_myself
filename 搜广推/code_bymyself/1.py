from asyncio import QueueEmpty
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que = deque()
        result = []
        
        print(f"Initial que: {list(que)}")  # 打印初始化的 que
        
        for i in range(len(nums)):
            if que and que[0] < i- k + 1:
                que.popleft()
            while que and nums[i] > nums[que[-1]]:
                que.pop()
                
            que.append(i)
            if i-k+1>=0:
                result.append(nums[que[0]])
            
        return result

# 测试案例
solution = Solution()
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print("Output:", solution.maxSlidingWindow(nums, k))

