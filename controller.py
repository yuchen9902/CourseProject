

#Reference:http://hplgit.github.io/web4sciapps/doc/pub/._part0004_web4sa_flask.html

from flask import Flask, render_template, request
from model import InputForm
import load_data
import classifier
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

app = Flask(__name__)

@app.route('/review', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        r = form.r.data
        train_dir = 'data/train'
        test_dir = 'data/test2'
        test_data,test_label,train_data,train_label = load_data.loadData(train_dir, test_dir)
        test_data = tokenizer.tokenize(r)
        predictor = classifier.unigram(train_data,train_label,test_data)
        if predictor[0]==0:
            s="This is a Negative movie review!"

        else:
            s="This is a Positive movie review!"

    else:
        s = None

    return render_template("view.html", form=form, s=s)


if __name__ == '__main__':
    app.run(debug=True)