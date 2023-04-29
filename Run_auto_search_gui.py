import subprocess
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import os
#import nltk
#from nltk.tokenize import word_tokenize
#from nltk import pos_tag
#from python_function import *





from tkinter import *
import random
import threading
import time
root = Tk()
root.geometry("600x800")
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import tkinter as tk




#a function for removing pronouns from a string
from nltk import pos_tag
from nltk.tokenize import word_tokenize



import re

import os






class ChatApp(QWidget):
    #global line_scores
    def __init__(self):
        super().__init__()
        
        # Define the maximum number of results to be printed
        self.max_print_num = 5
        
        #defining varables
        self.line_scores = {}
        
        # Set the window properties
        self.setWindowTitle("Chat App")
        self.setGeometry(100, 100, 600, 400)

        # Create the UI components
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #36393f;
                color: #dcddde;
                font-size: 14px;
            }
        """)
        
        self.chat_input = QLineEdit()
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: none;
                background-color: #40444b;
                color: #dcddde;
                font-size: 14px;
                padding: 10px;
            }
        """)
        
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: #7289da;
                color: #dcddde;
                font-size: 14px;
                padding: 10px;
            }

            QPushButton:hover {
                background-color: #677bc4;
            }
        """)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: #7289da;
                color: #dcddde;
                font-size: 14px;
                padding: 10px;
            }

            QPushButton:hover {
                background-color: #677bc4;
            }
        """)

        # Set the layout of the UI components
        self.query_file("university book deep Learning drew highs thy sdhe","1.txt")
        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addWidget(self.chat_input)
        self.chat_input.returnPressed.connect(self.send_message)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect the send button to the send message function
        self.send_button.clicked.connect(self.send_message)

        # Connect the run button to the run program function
        self.run_button.clicked.connect(self.run_program)
        
    def preprocess(self, line):
        # Remove punctuation and convert to lowercase
        line = line.translate(str.maketrans('', '', string.punctuation)).lower()
        # Tokenize the line into words
        words = nltk.word_tokenize(line)
        # Filter the words to only include nouns
        nouns = [word for word in words if self.is_noun(word)]
        # Join the nouns into a single string
        noun_string = ' '.join(nouns)
        return noun_string
        
    def update_scores(self,line_scores, line_num, score):
        """
        Updates the scores dictionary with the given line number and score.
        If the line number already exists in the dictionary, their score is updated.
        """
        if line_num in self.line_scores:
            self.line_scores[line_num] += score
        else:
            self.line_scores[line_num] = score
        
    def remove_pronouns(self, text):
        pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','re','ah','buzzer','oh','nt',"'s","'ll","n't"]
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        filtered_words = [word for word, tag in tagged_words if tag != 'PRP' and word.lower() not in pronouns]
        return ' '.join(filtered_words)
        
    def is_noun(self, word):
        pos = nltk.pos_tag([word])
        return pos[0][1] == 'NN'
        
        
    def query_file(self, query, ref_file):
        #self.update_chat("<p style=color:purple; font-weight:bold;>"+"They Spoke: "+ query + "<br>"+"</span>")
        #self.update_chat("<p style=color:purple; font-weight:bold;> They Spoke: "+ query + "<br> </span>")
        self.line_scores = {}
        max_print_num = self.max_print_num
        print_num = 0
        query = self.remove_pronouns(query)
        query_nouns = self.preprocess(query)
        query_nouns_list = query_nouns.split()
        # Define the number of sentences to include before and after the relevant sentence
        num_context_sentences = 2

        # Open the text file for reading
        with open(ref_file, 'r') as file:
            # Read the file line by line
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Preprocess the line to only include nouns
                noun_string = self.preprocess(line)
                noun_string_list = noun_string.split()
                score = 0
                for item in query_nouns_list:
                    # Check if the item is in list2
                    if item in noun_string_list:
                        score = score+1
                        # Update the values of the line scores in the line scores dictionary
                        self.update_scores(self.line_scores, i, score)
            #for value in self.line_scores.values():
            #    print(value)
            self.line_scores = dict(self.sort_scores(self.line_scores))
            for line_num in self.line_scores.items():
                #print(line_num)
            #print(self.line_scores)
            #result = ""
            #print(self.line_scores.items())
            #for line_num in self.line_scores.items():
                #print(line_num)
                if not print_num >= max_print_num:

                    j = int(line_num[0])
                    start_index = max(0, j - num_context_sentences)
                    end_index = min(len(lines), j + num_context_sentences + 1)
                    context = ''.join(lines[start_index:end_index])
                    for itemz in query_nouns_list:
                        print(itemz)
                        context = self.capitalize_and_bold_blue_word(context, itemz)
                    #result += context + "\n"
                    print_num = print_num+1
                    print(context)
                    print()
                    self.update_chat("Bot: "+ context)
            #return result

    #def capitalize_and_underline_word(self, string, word):
    #    string = re.sub(re.compile(word, re.IGNORECASE), '\033[4m' + word.upper() + '\033[0m', string)
    #    return string
        
    def capitalize_and_underline_word(self, string, word):
        # Check if the word is already capitalized in the string
        if word.upper() in string:
            return string
        
        # Capitalize the word and underline it
        string = re.sub(re.compile(word, re.IGNORECASE), '\033[4m' + word.upper() + '\033[0m', string)
        return string
        
    def capitalize_and_bold_blue_word(self, string, word):
        # Check if the word is already capitalized in the string
        if word.upper() in string:
            return string
        
        # Capitalize the word and format it as bold and blue
        #string = re.sub(re.compile(word, re.IGNORECASE), '\033[1m\033[34m' + word.upper() + '\033[0m', string)
        string = re.sub(re.compile(word, re.IGNORECASE), f'<span style="color: #87CEFA; font-weight: bold;">{word.upper()}</span>', string)
        return string


    def sort_scores(self, scores_dict):
        """
        Sorts the scores dictionary in descending order based on the scores.
        """
        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def send_message(self):
        string_message = self.chat_input.text()
        message = self.format_bold_white(string_message)
        self.chat_history.append(f"<b>You:</b> {message}")
        #self.chat_history.append("")
        self.chat_input.clear()
        print(string_message)
        self.query_file(str(string_message), "1.txt")
        print(string_message)
        #self.query_file("university book deep Learning drew highs thy sdhe","1.txt")

    def run_program(self):
        self.chat_history.append("<b>Program started</b>")
        self.chat_history.append("")
        cwd = os.getcwd()
        self.thread = threading.Thread(target=self.execute_program)
        self.thread.start()
        
    def format_bold_white(self, string):
        return f'<span style="font-weight: bold; color: white;">{string}</span>'


    def execute_program(self):
        self.program_process = subprocess.Popen(["./stream", "-f", "output"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = self.program_process.stdout.readline().decode()
            #self.update_chat("<p style=color:purple; font-weight:bold;> They Spoke: "+ output + "<br> </span>")
            query_output = self.query_file(output, "1.txt")
            #self.update_chat("<p style=color:purple; font-weight:bold;> They Spoke: "+ output + "<br> </span>")
            #self.query_file(output, "1.txt")
            if not query_output == "Nullz":
                    #if query_output == "Nullz":
                    #    time.sleep(2)
                    #    self.execute_program()
                    #    #self.update_chat("Bot: "+ "Nullz")
                    #    print("Nullz")
		        
                    if not output and self.program_process.poll() is not None:
                    	break
                    #self.update_chat("Bot: "+ output)
                    #self.update_chat("Bot: "+ query_output)
        self.chat_history.append("<b>Program finished</b>")
        self.chat_history.append("")

    def update_chat(self, message):
        self.chat_history.append("<br>")
        #self.chat_history.insertHtml("<span style='color: blue; font-weight: bold;'>" + message + "</span><br>")
        self.chat_history.insertHtml(message)
        


    def closeEvent(self, event):
            # Stop the thread if it's running
        if hasattr(self, "program_thread") and self.program_thread.is_alive():
            self.program_thread.do_run = False
            self.program_thread.join()
    
            # Terminate the subprocess if it's running
        if hasattr(self, "program_process") and self.program_process.poll() is None:
            self.program_process.terminate()
    
        event.accept()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #2f3136;
        }
    """)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
    #sys.exit()