def trim(s):
    for i in s:
        if i== " ":
            s = s[1:]
        else:
            break
    for k in s[::-1]:
        if k == " ":
            s = s[:-1]
        else:
            break
    return s

if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')
