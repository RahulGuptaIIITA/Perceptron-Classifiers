import re
import sys
import json
import string


class PercepClassify:
    
    def __init__(self, model):
        
        self.count = 1
        self.class1_bias = 0
        self.class2_bias = 0
        self.class1_beta = 0
        self.class2_beta = 0
        self.class1_weight = dict()
        self.class2_weight = dict()
        self.class1_cached_weight = dict()
        self.class2_cached_weight = dict()
        self.file_name = model.rsplit("/", 1)
        self.file_name = self.file_name[len(self.file_name)-1]
        
        if self.file_name == "averagedmodel.txt":
            with open(model) as file:
                lines = file.readlines()
                for line in lines:
                    var_obj = json.loads(line)
                    
                    if var_obj['type'] == "count":
                        self.count = var_obj['value']
                        
                    elif var_obj['type'] == "bias":
                        self.class1_bias = var_obj['class1']
                        self.class2_bias = var_obj['class2']
                    
                    elif var_obj['type'] == "beta":
                        self.class1_beta = var_obj['class1']
                        self.class2_beta = var_obj['class2']
                    
                    elif var_obj['type'] == "weight":
                        self.class1_weight = var_obj['class1']
                        self.class2_weight = var_obj['class2']
                    
                    elif var_obj['type'] == "cached_weight":
                        self.class1_cached_weight = var_obj['class1']
                        self.class2_cached_weight = var_obj['class2']

        elif self.file_name  == "vanillamodel.txt":
            with open(model) as file:
                lines = file.readlines()
                for line in lines:
                    var_obj = json.loads(line)
                    if var_obj['type'] == "count":
                        self.count = var_obj['value']
                        
                    elif var_obj['type'] == "bias":
                        self.class1_bias = var_obj['class1']
                        self.class2_bias = var_obj['class2']
    
                    elif var_obj['type'] == "weight":
                        self.class1_weight = var_obj['class1']
                        self.class2_weight = var_obj['class2']
                    
        return
    
    def classify_sentence(self, sentence, model, file_object):
        sentence = sentence.strip()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = sentence.split(" ")
        
        hash = tokens[0]
        words = tokens[1:]
        words = [word.lower().strip() for word in words]

        class1_activation = 0
        class2_activation = 0
        unique_word_freq_dict = dict()
        for word in words:
            if word not in unique_word_freq_dict:
                unique_word_freq_dict[word] = 0
            unique_word_freq_dict[word] += 1
        
        if self.file_name == "vanillamodel.txt":
            
            for word, freq in unique_word_freq_dict.iteritems():
                if word in self.class1_weight:
                    class1_activation += self.class1_weight[word] * freq
    
                if word in self.class2_weight:
                    class2_activation += self.class2_weight[word] * freq

            class1_activation += self.class1_bias
            class2_activation += self.class2_bias
            
        elif self.file_name == "averagedmodel.txt":

            for word, freq in unique_word_freq_dict.iteritems():
                if word in self.class1_weight:
                    if word in self.class1_cached_weight:
                        class1_activation += (self.class1_weight[word] - self.class1_cached_weight[word]/float(self.count)) * freq
        
                if word in self.class2_weight:
                    if word in self.class2_cached_weight:
                        class2_activation += (self.class2_weight[word] - self.class2_cached_weight[word] / float(self.count)) * freq

            class1_activation += (self.class1_bias - self.class1_beta/float(self.count))
            class2_activation += (self.class2_bias - self.class2_beta/float(self.count))
            
        class1 = "True" if class1_activation >= 0 else "Fake"
        class2 = "Pos"  if class2_activation >= 0 else "Neg"
        
        file_object.write(hash + " " + class1 + " " + class2 + "\n")
        
        return
    
    def run(self, infile, model):
        try:
            file_object = open("percepoutput.txt", "w")
            with open(infile) as file:
                sentences = file.readlines()
                for sentence in sentences:
                    self.classify_sentence(sentence, model, file_object)
            
            file_object.close()
        
        except Exception as e:
            print (e)
        
        return


if __name__ == "__main__":
    infile_model = sys.argv[1]
    infile_classify = sys.argv[2]
    #infile_model = "averagedmodel.txt"
    #infile_model = "vanillamodel.txt"
    #infile_classify = "data/dev-text.txt"
    percep_classify_object = PercepClassify(infile_model)
    percep_classify_object.run(infile_classify, infile_model)

