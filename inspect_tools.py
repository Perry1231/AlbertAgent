from tools import ALL_TOOLS

print(f"TOOLS COUNT: {len(ALL_TOOLS)}")
print("=" * 60)

for tool in ALL_TOOLS:
    print(f"NAME: {tool.name}")
    print(f"DESCRIPTION: {tool.description}")
    print(f"INPUTS: {tool.inputs}")
    print("-" * 60)