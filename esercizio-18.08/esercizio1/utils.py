def open_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return "Error"
    
def read_lines(file_path: str) -> int:

    opened_file = open_file(file_path)
    print(opened_file)
    lines = opened_file.splitlines()
    print(lines)
    return len(lines)
    
print(read_lines("prova.txt"))