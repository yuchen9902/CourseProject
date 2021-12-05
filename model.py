from wtforms import Form, StringField, validators

class InputForm(Form):
    r = StringField(validators=[validators.InputRequired()])