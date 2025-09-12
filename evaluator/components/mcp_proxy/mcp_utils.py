import re

def clean_tool_name(name):
    # Replace spaces and dots with underscores
    name = name.replace(' ', '_').replace('.', '_')
    # Remove all non-ASCII characters (unicode)
    name = re.sub(r'[^\x00-\x7F]+', '', name)
    # Remove leading underscores
    name = name.lstrip('_')

    return name
