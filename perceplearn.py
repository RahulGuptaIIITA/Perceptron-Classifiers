import re
import sys
import json
import string


class PercepLearn:
    
    def __init__(self):
        self.count = 1
        self.class1_bias = 0
        self.class2_bias = 0
        self.class1_beta = 0
        self.class2_beta = 0
        self.class1_weight = dict()
        self.class2_weight = dict()
        self.class1_cached_weight = dict()
        self.class2_cached_weight = dict()
        self.sentence_word_dict = []
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                'until', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
                 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
                 'us', 'much', 'would', 'either', 'indeed', 'seems']
        
        return

    def save_model(self):
    
        vanilla_object = open("vanillamodel.txt", "w")
        obj = {
            'type': 'count',
            'value': self.count
        }
        vanilla_object.write(json.dumps(obj) + "\n")
        
        obj = {
            'type': 'bias',
            'class1': self.class1_bias,
            'class2': self.class2_bias
        }
        vanilla_object.write(json.dumps(obj) + "\n")
        
        obj = {
            'type': 'weight',
            'class1': self.class1_weight,
            'class2': self.class2_weight
        }
        vanilla_object.write(json.dumps(obj) + "\n")

        average_object = open("averagedmodel.txt", "w")
        obj = {
            'type': 'count',
            'value': self.count
        }
        average_object.write(json.dumps(obj) + "\n")
        obj = {
            'type': 'bias',
            'class1': self.class1_bias,
            'class2': self.class2_bias
        }
        average_object.write(json.dumps(obj) + "\n")

        obj = {
            'type': 'beta',
            'class1': self.class1_beta,
            'class2': self.class2_beta
        }
        average_object.write(json.dumps(obj) + "\n")
        
        obj = {
            'type': 'weight',
            'class1': self.class1_weight,
            'class2': self.class2_weight
        }
        average_object.write(json.dumps(obj) + "\n")
        
        obj = {
            'type': 'cached_weight',
            'class1': self.class1_cached_weight,
            'class2': self.class2_cached_weight
        }
        average_object.write(json.dumps(obj) + "\n")
        
        return
    
    def calculate_activation(self):
        for value in self.sentence_word_dict:
            (l1, l2, words) = value
            class1_activation = 0
            class2_activation = 0
            for word, freq in words.iteritems():
                if word in self.class1_weight:
                    class1_activation += self.class1_weight[word] * freq
              
                if word in self.class2_weight:
                    class2_activation += self.class2_weight[word] * freq
            
            class1_activation += self.class1_bias
            class2_activation += self.class2_bias

            y1 = (1 if l1 == "True" else -1)
            y2 = (1 if l2 == "Pos" else -1)
            
            class1_activation = class1_activation * y1
            class2_activation = class2_activation * y2
 
            
            if class1_activation <= 0:
                self.class1_bias += y1
                self.class1_beta += y1 * self.count

                for word, freq in words.iteritems():
                    if word in self.class1_weight:
                        self.class1_weight[word] += freq * y1
                    else:
                        self.class1_weight[word] = freq * y1

                    if word in self.class1_cached_weight:
                        self.class1_cached_weight[word] += freq * self.count * y1
                    else:
                        self.class1_cached_weight[word] = freq * self.count * y1
            
            if class2_activation <= 0:
                self.class2_bias += y2
                self.class2_beta += y2 * self.count
    
                for word, freq in words.iteritems():
                    if word in self.class2_weight:
                        self.class2_weight[word] += freq * y2
                    else:
                        self.class2_weight[word] = freq * y2
        
                    if word in self.class2_cached_weight:
                        self.class2_cached_weight[word] += freq * self.count * y2
                    else:
                        self.class2_cached_weight[word] = freq * self.count * y2
        
            self.count += 1

        return
    
    def parse_sentence(self, sentence):
        sentence = sentence.strip()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = sentence.split(" ")
        
        label1 = tokens[1]
        label2 = tokens[2]
        words = tokens[3:]

        unique_word_freq_dict = dict()
        words = [word.lower().strip() for word in words if word.lower() not in self.stop_words]
        
        for word in words:
            if word not in unique_word_freq_dict:
                unique_word_freq_dict[word] = 0
            unique_word_freq_dict[word] += 1
        
        self.sentence_word_dict.append((label1, label2, unique_word_freq_dict))
        
        return
    
    def run(self, infile):
        try:
            with open(infile) as file:
                sentences = file.readlines()
                for sentence in sentences:
                    self.parse_sentence(sentence)
            
            for i in range(0, 30):
                self.calculate_activation()
            
            self.save_model()
            print self.count
        except Exception as e:
            print (e)
        
        return


if __name__ == "__main__":
    infile = sys.argv[1]
    percep_learn_object = PercepLearn()
    percep_learn_object.run(infile)
    #percep_learn_object.run("small_data")
    #percep_learn_object.run("data/train-labeled.txt")
