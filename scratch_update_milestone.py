with open(".ai/MILESTONE_STATUS.md", "r", encoding="utf-8") as f:
    content = f.read()

m6_section = '''
## M6 API Modernization
- [x] M6.1 API & Service Layer Architecture Modernization
- [x] M6.2 API Contracts, Request Validation & Response Models
'''
content = content + m6_section

with open(".ai/MILESTONE_STATUS.md", "w", encoding="utf-8") as f:
    f.write(content)
