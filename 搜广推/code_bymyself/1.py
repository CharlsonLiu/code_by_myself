from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode(next=head)
        cur = dummy
        start, end = dummy.next, dummy.next
        n = 1
        
        while end:
            if n < k:
                end = end.next
                n += 1
            else:
                temp = end.next
                cur.next = self.reverseList(start, end)
                start.next = temp
                cur = start
                start = temp
                end = temp
                n = 1
        return dummy.next

    def reverseList(self, head: Optional[ListNode], end: Optional[ListNode]) -> Optional[ListNode]:
        pre, cur = None, head
        while cur != end:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        cur.next = pre  # 最后一个节点也需要指向前一个节点
        self.printList(cur)  # 打印反转后的链表
        return cur

    def printList(self, node: Optional[ListNode]) -> None:
        """打印链表的辅助函数"""
        values = []
        while node:
            values.append(node.val)
            node = node.next
        print(" -> ".join(map(str, values)))

def create_linked_list(arr):
    """创建链表的辅助函数"""
    dummy = ListNode()
    current = dummy
    for value in arr:
        current.next = ListNode(value)
        current = current.next
    return dummy.next

def linked_list_to_list(head):
    """将链表转换为列表的辅助函数"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

# 测试用例
def test_reverse_k_group():
    solution = Solution()
    
    # 测试用例 1
    head1 = create_linked_list([1, 2, 3, 4, 5])
    k1 = 2
    result1 = solution.reverseKGroup(head1, k1)
    print("测试用例 1 结果:", linked_list_to_list(result1))  # 预期输出: [2, 1, 4, 3, 5]

    # 测试用例 2
    head2 = create_linked_list([1, 2, 3, 4, 5])
    k2 = 3
    result2 = solution.reverseKGroup(head2, k2)
    print("测试用例 2 结果:", linked_list_to_list(result2))  # 预期输出: [3, 2, 1, 4, 5]

    # 测试用例 4
    head4 = create_linked_list([1, 2])
    k4 = 2
    result4 = solution.reverseKGroup(head4, k4)
    print("测试用例 4 结果:", linked_list_to_list(result4))  # 预期输出: [1, 2]

# 运行测试
test_reverse_k_group()
