import load_data
import classifier
import numpy
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
def main():
    #For running on provided test data
    train_dir = 'data/train'
    test_dir = 'data/test2'

    # test_data,test_label,train_data,train_label = load_data.loadData(train_dir, test_dir)
    
    # predictor = classifier.unigram(train_data,train_label,test_data)
    # accuracy = numpy.mean(predictor==test_label)
    # print(accuracy)

    #For running on customized data:
    test_data,test_label,train_data,train_label = load_data.loadData(train_dir, test_dir)
    test_string = "There are certain horror directors for whom I've built up so much respect & admiration over the years, that they can't possibly disappoint me know matter what garbage to decide."
    test_data = tokenizer.tokenize(test_string)
    predictor = classifier.unigram(train_data,train_label,test_data)
    if predictor[0]==0:
        print("This is a Negative movie review!")
    else:
        print("This is a Positive movie review!")
if __name__ == "__main__":
    main()
