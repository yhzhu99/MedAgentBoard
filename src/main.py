def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(5)]
print(result)
     