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
    lines = opened_file.splitlines()
    return len(lines)
    
print(read_lines("prova.txt"))

def total_words(file_path: str) -> int:
    
    opened_file = open_file(file_path)
    words = opened_file.split()
    return len(words)

print(total_words("prova.txt"))

def top_five(file_path: str) -> list:
    
    opened_file = open_file(file_path)
    words = opened_file.split()
    word_count = {}
    
    for word in words:
        word = word.lower().strip('.,!?";:')
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
            
    sorted_words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    top_five_words = [word for word, count in sorted_words[:5]]
    
    return top_five_words

print(top_five("prova.txt"))