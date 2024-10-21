n = int(input())
day = 0
while n>4:
    n = n-(n//2+2)
    day+=1
print(day+1)
