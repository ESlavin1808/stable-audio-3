import re, sys
with open(r'F:\XTTS\zxcvb2521\stable-audio-3\app\static\index.html', 'r', encoding='utf-8') as f:
    content = f.read()
m = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
if not m:
    print('No script section found')
    sys.exit(1)
js = m.group(1)
# Check balanced braces
opens = js.count('{')
closes = js.count('}')
print(f'Braces: open={{ {opens}, close=}} {closes}')
if opens == closes:
    print('Balanced braces OK')
else:
    print(f'Unbalanced! diff={opens-closes}')
# Check balanced parens
opens = js.count('(')
closes = js.count(')')
if opens == closes:
    print('Balanced parens OK')
else:
    print(f'Unbalanced parens! diff={opens-closes}')
