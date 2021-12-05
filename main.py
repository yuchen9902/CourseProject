import load_data
import classifier
def main():
    train_dir = 'data/train'
    test_dir = 'data/test'

    test_data,test_label,train_data,train_label = load_data.loadData(train_dir, test_dir)
    print(test_data)
if __name__ == "__main__":
    main()
