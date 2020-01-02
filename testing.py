def addition(n):
    return n + n


numbers = (1, 2, 3, 4)

result = map(addition, numbers)

list_result = list(result)
print(list_result)
print(type(list_result))
column_list = list_result.reshape(-1, 1)
print(column_list)
