import joblib
import flask as fl
#from flask import Flask, render_template, redirect, url_for, request
import flask_bootstrap as fb
#from flask_bootstrap import Bootstrap
import flask_wtf as fw
#from flask_wtf import FlaskForm
import wtforms as wtf
#from wtforms import StringField, PasswordField, BooleanField
#from wtforms.validators import InputRequired, Email, Length
import flask_sqlalchemy as fsq
#from flask_sqlalchemy import SQLAlchemy
#import werkzeug as we
#from werkzeug.security import generate_password_hash, check_password_hash
import flask_login as flg
#from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
import sklearn
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))

app = fl(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = fb.Bootstrap(app)
db = fsq.SQLAlchemy(app)

login_manager = fl.LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# KEY:3123-1664284021

class User(fl.UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(fw.FlaskForm):
    username = wtf.StringField('Username', validators=[wtf.InputRequired(), wtf.Length(min=4, max=15)])
    password = wtf.PasswordField('Password', validators=[wtf.InputRequired(), wtf.Length(min=8, max=80)])
    remember = wtf.BooleanField('remember me')


class RegisterForm(fw.FlaskForm):
    email = wtf.StringField('Email', validators=[wtf.InputRequired(), wtf.Email(message='Invalid email'), wtf.Length(max=50)])
    username = wtf.StringField('Username', validators=[wtf.InputRequired(), wtf.Length(min=4, max=15)])
    password = wtf.PasswordField('Password', validators=[wtf.InputRequired(), wtf.Length(min=8, max=80)])


@app.route('/')
def index():
    return fl.render_template("index.html")


@app.route('/about')
def about():
    return fl.render_template("about.html")


@app.route('/help')
def help():
    return fl.render_template("help.html")


@app.route('/terms')
def terms():
    return fl.render_template("tc.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if (form.password.data):
                flg.login_user(user, remember=form.remember.data)
                return fl.redirect(fl.url_for('dashboard'))

        return fl.render_template("login.html", form=form)
    return fl.render_template("login.html", form=form)

# KEY:3123-1664284021

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.create_all()
        db.session.add(new_user)
        db.session.commit()

        return fl.redirect("/login")
    return fl.render_template('signup.html', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return fl.render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return fl.render_template("disindex.html")



@app.route("/kidney")
@login_required
def kidney():
    return fl.render_template("kidney.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if fl.request.method == "POST":
        to_predict_list = fl.request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return fl.render_template("kidney_result.html", prediction_text=prediction)





@app.route('/logout')
@login_required
def logout():
    flg.logout_user()
    return fl.redirect(fl.url_for('index'))



# KEY:3123-1664284021
##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = sklearn.RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################




############################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

