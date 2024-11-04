import os
from tqdm import tqdm
import random
from math import log, exp
import pickle


class NGramModel:
    def __init__(self):
        self.total_unigrams = 0
        self.total_bigrams = 0
        self.total_trigrams = 0

        # Necessary for good-turing smoothing
        self.unigram_frequency_counts = {}
        self.bigram_frequency_counts = {}
        self.trigram_frequency_counts = {}
        

        # dictionary for holding unigram counts
        self.unigram_counts = {}
        # dictionary for holding biigram counts
        self.bigram_counts = {}
        # dictionary for holding trigram counts
        self.trigram_counts = {}

        self.token_to_punctutation = {
            '<period>': '.', 
            '<new_line>': '\n', 
            '<space>': ' ',
            '<exclamation_mark>': '!',
            '<dash>': '-',
            '<question_mark>' : '?',
            '<back_slash>' : '\\',
            '<forward_slash>' : '/',
            '<double_quotes>' : '"',
            '<single_quotes>' : "'",
            '<semicolon>' : ";",
            '<column>' : ":",
            '<open_paranthesis>' :"(",
            '<close_paranthesis>' :")",
            '<open_square_bracket>' :"[",
            '<close_square_bracket>' :"]",
            '<open_curly_bracket>' : "{",
            '<close_curly_bracket>' : "}",
            '<percentage_symbol>': "%"
        }


    

    # Trains the model and save the models stats on given output_folder
    def train_model(self, input_folder):

        unigram_counts_input_file = os.path.join(input_folder, 'unigram.txt')
        bigram_counts_input_file = os.path.join(input_folder, 'bigram.txt')
        trigram_counts_input_file = os.path.join(input_folder, 'trigram.txt')


        self._count_unigrams(unigram_counts_input_file)
        self._count_bigrams(bigram_counts_input_file)
        self._count_trigrams(trigram_counts_input_file)



        for _, value in self.unigram_counts.items():
            if value in self.unigram_frequency_counts:
                self.unigram_frequency_counts[value] += 1
            else:
                self.unigram_frequency_counts[value] = 1
                


        for _, value in self.bigram_counts.items():
            if value in self.bigram_frequency_counts:
                self.bigram_frequency_counts[value] += 1
            else:
                self.bigram_frequency_counts[value] = 1


        for _, value in self.trigram_counts.items():
            if value in self.trigram_frequency_counts:
                self.trigram_frequency_counts[value] += 1                
            else:
                self.trigram_frequency_counts[value] = 1



    def save_model(self, file_path):
        """Saves the model data to a file for later use."""
        model_data = {
            'total_unigrams': self.total_unigrams,
            'total_bigrams': self.total_bigrams,
            'total_trigrams': self.total_trigrams,
            'unigram_frequency_counts': self.unigram_frequency_counts,
            'bigram_frequency_counts': self.bigram_frequency_counts,
            'trigram_frequency_counts': self.trigram_frequency_counts,
            'unigram_counts': self.unigram_counts,
            'bigram_counts': self.bigram_counts,
            'trigram_counts': self.trigram_counts
        }
        
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model saved to {file_path}")

    
    def load_model(self, file_path):
        """Loads model data from a file."""
        try:
            with open(file_path, 'rb') as file:
                model_data = pickle.load(file)
                
                self.total_unigrams = model_data.get('total_unigrams', 0)
                self.total_bigrams = model_data.get('total_bigrams', 0)
                self.total_trigrams = model_data.get('total_trigrams', 0)

                self.unigram_frequency_counts = model_data.get('unigram_frequency_counts', {})
                self.bigram_frequency_counts = model_data.get('bigram_frequency_counts', {})
                self.trigram_frequency_counts = model_data.get('trigram_frequency_counts', {})

                self.unigram_counts = model_data.get('unigram_counts', {})
                self.bigram_counts = model_data.get('bigram_counts', {})
                self.trigram_counts = model_data.get('trigram_counts', {})
                
            print(f"Model loaded from {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    
    def _count_unigrams(self, input_file):
        print('Counting the unigrams... This may take some time!')
        with open(input_file, 'r') as infile:
            for line in tqdm(infile):
                for word in line.split():
                    self._increment_unigram_count(word)
                    self.total_unigrams += 1
        print('Counting the unigrams has been completed!')

        

    def _increment_unigram_count(self, word):
        if word in self.unigram_counts:
            self.unigram_counts[word] += 1
        else:
            self.unigram_counts[word] = 1


    def _count_bigrams(self, input_file):
        print('Counting the bigrams... This may take some time!')
        with open(input_file, 'r') as infile:
            for line in tqdm(infile):
                w0, w1 = line.split()
                self._increment_bigram_count(w0, w1)
                self.total_bigrams += 1

        print('Counting the bigrams has been completed!')

        

    def _increment_bigram_count(self, w0, w1):
        key = (w0, w1)
        if key in self.bigram_counts:
            self.bigram_counts[key] += 1
        else:
            self.bigram_counts[key] = 1




    def _count_trigrams(self, input_file):
        print('Counting the trigrams... This may take some time!')
        with open(input_file, 'r') as infile:
            for line in tqdm(infile):
                w0, w1, w2 = line.split()
                self._increment_trigram_count(w0, w1, w2)
                self.total_trigrams += 1
        print('Counting the bigrams has been completed!')


    
    def _increment_trigram_count(self, w0, w1, w2):
        # Use a tuple as a key for the trigram
        key = (w0, w1, w2)
        if key in self.trigram_counts:
            self.trigram_counts[key] += 1
        else:
            self.trigram_counts[key] = 1



    def _get_count_smoothed(self, key, n):
        if n == 1:  # unigram
            w0 = key[0]
            if w0 in self.unigram_counts:
                c = self.unigram_counts[w0]
                N_c = self.unigram_frequency_counts[c]

                if c+1 in self.unigram_frequency_counts:
                    N_c_plus_1 = self.unigram_frequency_counts[c+1]
                else:
                    N_c_plus_1 = 1
                return (c + 1) * N_c_plus_1 / N_c
            else:
                return 1
        
        elif n == 2:  # bigram
            if key in self.bigram_counts:
                c = self.bigram_counts[key]
                N_c = self.bigram_frequency_counts[c]
                if c+1 in self.bigram_frequency_counts:
                    N_c_plus_1 = self.bigram_frequency_counts[c+1]
                else:
                    N_c_plus_1 = 1
                return (c + 1) * N_c_plus_1 / N_c
            else:
                return 1
        
        else:  # trigram
            if key in self.trigram_counts:
                c = self.trigram_counts[key]
                N_c = self.trigram_frequency_counts[c]
                if c+1 in self.trigram_frequency_counts:
                    N_c_plus_1 = self.trigram_frequency_counts[c+1]
                else:
                    N_c_plus_1 = 1
                return (c + 1) * N_c_plus_1 / N_c
            else:
                return 1


    def _compute_probability(self, key, n):
        if n == 1:  # unigram
            return log(self._get_count_smoothed(tuple(key), 1) / self.total_unigrams)
        elif n == 2:  # bigram
            return log(self._get_count_smoothed(tuple(key), 2) / self._get_count_smoothed((key[0],), 1))
        else:  # trigram
            return log(self._get_count_smoothed(tuple(key), 3) / self._get_count_smoothed((key[0], key[1]), 1))




    def _get_most_probable_unigrams(self, m):
        l = []
        for _ in range(m):
            max_c = 0
            max_word = None
            for word, count in self.unigram_counts.items():
                if word in l:
                    continue
                if count > max_c:
                    max_word = word
                    max_c = count
            if max_word is not None:
                l.append(max_word)
            else:
                break

        return l



    def _get_most_probable_bigrams(self, m, key):
        l = []
        w0 = key[1]
        for _ in range(m):
            max_c = 0
            max_w1 = None
            for key, val in self.bigram_counts.items():
                _w0, _w1 = key
                if _w1 in l or w0 != _w0:
                    continue
                if val > max_c:
                    max_w1 = _w1
                    max_c = val
            if max_w1 is not None:
                l.append(max_w1)
            else:
                break
        
        return [[w0, w1] for w1 in l]




    def _get_most_probable_trigrams(self, m, key):
        l = []
        w0, w1 = key[1:]
        for _ in range(m):
            max_c = 0
            max_w2 = None
            for key, val in self.trigram_counts.items():
                _w0, _w1, _w2 = key
                if _w2 in l or w0 != _w0 or w1 != _w1:
                    continue
                if val > max_c:
                    max_w2 = _w2
                    max_c = val
            if max_w2 is not None:
                l.append(max_w2)
            else:
                break
        
        return [[w0, w1, w2] for w2 in l]




    def _generate_sentence_unigram(self, n, max_len):
        counter = 0
        while True:
            l = self._get_most_probable_unigrams(n)
            if len(l) > 1:
                index = random.randint(0, len(l)-1)
            else:
                index = 0
            w0 = l[index]

            counter += 1
            if counter == max_len:
                break

            if w0 == '<end>':
                break

            if w0 in self.token_to_punctutation:
                print(self.token_to_punctutation[w0], end='')
            elif w0 == '<start>':
                continue
            else:
                print(w0, end='')





    def _generate_sentence_bigram(self, n, max_len):
        counter = 0
        keys_list = [key for key in self.bigram_counts.keys() if key[0] == '<start>']
        key = keys_list[random.randint(0, len(keys_list) - 1)]
        while True:
            l = self._get_most_probable_bigrams(n, key)
            if len(l) > 1:
                index = random.randint(0, len(l)-1)
            else:
                index = 0
            key = l[index]

            w1 = key[1]

            counter += 1
            if counter == max_len:
                break

            if w1 == '<end>':
                break

            if w1 in self.token_to_punctutation:
                print(self.token_to_punctutation[w1], end='')
            elif w1 == '<start>':
                continue
            else:
                print(w1, end='')





    def _generate_sentence_trigram(self, n, max_len):
        counter = 0

        keys_list = [key for key in self.bigram_counts.keys() if key[0] == '<start>']
        random_bigram = keys_list[random.randint(0, len(keys_list) - 1)]
        key = ['<start>', '<start>', random_bigram[1]]
        while True:
            l = self._get_most_probable_trigrams(n, key)
            if len(l) > 1:
                index = random.randint(0, len(l)-1)
            else:
                index = 0 
            key = l[index]

            w2 = key[2]

            counter += 1
            if counter == max_len:
                break

            if w2 == '<end>':
                break

            if w2 in self.token_to_punctutation:
                print(self.token_to_punctutation[w2], end='')
            elif w2 == '<start>':
                continue
            else:
                print(w2, end='')




    def generate_sentence(self, n, num_probables=5, max_len=100):
        if n == 1: # generate sentence using unigram
            self._generate_sentence_unigram(num_probables, max_len)
        elif n == 2: # generate sentence using bigram
            self._generate_sentence_bigram(num_probables, max_len)
        else: # generate sentence using trigram
            self._generate_sentence_trigram(num_probables, max_len)
        
        print('')




    def compute_perplexity(self, input_file, n):
        if n == 1:
            return self._compute_perplexity_unigram(input_file)
        elif n == 2:
            return self._compute_perplexity_bigram(input_file)
        else:
            return self._compute_perplexity_trigram(input_file)


    def _compute_perplexity_unigram(self, input_file):
        p = 0.0
        N = 0
        print("Computing perplexity...")
        with open(input_file, 'r') as input_file:
            for line in tqdm(input_file):
                word = line.split()[0]
                p += self._compute_probability([word], 1)
                N += 1
        return exp(-p / N)
    

    def _compute_perplexity_bigram(self, input_file):
        p = 0.0
        N = 0
        print("Computing perplexity...")
        with open(input_file, 'r') as input_file:
            for line in tqdm(input_file):
                bigram = line.split()
                p += self._compute_probability(bigram, 2)
                N += 1
        return exp(-p / N)
    


    def _compute_perplexity_trigram(self, input_file):
        p = 0.0
        N = 0
        print("Computing perplexity...")
        with open(input_file, 'r') as input_file:
            for line in tqdm(input_file):
                trigram = line.split()
                p += self._compute_probability(trigram, 3)
                N += 1
        return exp(-p / N)
