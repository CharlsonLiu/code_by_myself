class Solution:
    def numOfSubarrays(self, arr):
        # 构建前缀和数组
        l = len(arr)
        s = [0] * (l + 1)
        for i, a in enumerate(arr):
            s[i+1] = s[i] + a
        
        odd_count = 0
        even_count = 1 # 包括前缀和0 
        result = 0
        dp = [0] * (l + 1)
        for i in range(1, l + 1):
            if s[i] % 2 == 1:
                dp[i] += even_count
                odd_count +=1
            else:
                dp[i] += odd_count
                even_count +=1
            result += dp[i]
        
        # 打印 dp 数组
            print("DP array:", dp)
        return dp[-1]

# 测试代码
solution = Solution()
arr = [1,2,3,4,5,6,7]
solution.numOfSubarrays(arr)
