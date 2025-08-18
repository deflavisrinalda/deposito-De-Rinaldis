from utils import *

def main():

    #numero totale righe
    print(read_lines("prova.txt"))

    #numero totale parole
    print(total_words("prova.txt"))
    
    #top5 parole più frequenti
    print(top_five("prova.txt"))


if __name__ == "__main__":
    main()
    # This is a simple Python script that prints a greeting message.
    # It serves as an entry point for the program.