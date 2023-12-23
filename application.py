from flask import Flask, render_template, request, redirect, Response, url_for, session, flash
from flask_session import Session
from PIL import Image

import nltk
import os
import base64
import io
import datetime
import json
import random
import joblib
import pickle
import cv2

#Password Hashing
from flask_bcrypt import Bcrypt

#Import ML
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import load_model
from keras_metrics import f1_score
from keras.preprocessing import image

# Database
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Integer, String, and_, func


class Base(DeclarativeBase):
  pass

db = SQLAlchemy(model_class=Base)


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
db.init_app(app)
bcrypt = Bcrypt(app)

#session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(256), nullable=False)
    history = db.relationship("History", backref="user")
    review = db.relationship("Review", backref="user")
    
    def __repr__(self):
        return f'<User(id={self.id}, fullname={self.fullname})>'
    
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    def __repr__(self):
        return f'<History(id={self.id}, result={self.result})>'

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    def __repr__(self):
        return f'<Review(id={self.id}, review={self.review})>'
    
with app.app_context():
    db.create_all()
    
#Image Function
def facecrop(image):
    facedata = os.path.join(os.path.dirname(__file__), 'static/haarcascades/haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        final = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fname+ext, final)

    return


# Preparing and pre-processing the image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(48, 48, 3))
    # img = op_img.resize((48, 48))
    # img = np.array(img)
    img = np.expand_dims(img, axis=0)
    
    
    return img
 
 
# Predicting function
def predict_result(predict):
    model_image = load_model(os.path.join(os.path.dirname(__file__), 'models/citradigital/model.h5'), custom_objects={'f1_score': f1_score})
    pred = model_image.predict(predict)
    predicted = np.argmax(pred[0])
    
    classes=['fear', 'angry', 'disgust', 'happy', 'neutral', 'sad']
    
    print(pred)

    if classes[predicted] == 'fear':
        return 0
    elif classes[predicted] == 'angry':
        return 1
    elif classes[predicted] == 'disgust':
        return 2
    elif classes[predicted] == 'happy':
        return 3
    elif classes[predicted] == 'neutral':
        return 4
    else:
        return 5
#End Image Function

#Sentimen Function
model_sentimen = joblib.load(open(os.path.join(os.path.dirname(__file__), 'models/sentiment_analysis/svm_model.pkl'), 'rb'))
cv = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models/sentiment_analysis/cv.pickle'), 'rb'))



def predict_sentiment(test):
    test = [str(test)]
    test_vector = cv.transform(test).toarray()
    pred = model_sentimen.predict(test_vector)
    return pred[0]
#End Sentimen Function


#Chatbot Function
model_chatbot = load_model(os.path.join(os.path.dirname(__file__), 'models/chatbot/chatbot_model.h5'))
file_intents = os.path.join(os.path.dirname(__file__), 'models/chatbot/intents.json')
file_words = os.path.join(os.path.dirname(__file__), 'models/chatbot/words.pkl')
file_classes = os.path.join(os.path.dirname(__file__), 'models/chatbot/classes.pkl') 

intents = json.loads(open(file_intents).read())
words = pickle.load(open(file_words,'rb'))
classes = pickle.load(open(file_classes,'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model_chatbot.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
# End Chatbot Function

@app.route("/list", methods=['GET', 'POST'])
def list():
    title = "K-Genz | List"
    users = db.session.execute(db.select(User).order_by(User.id)).scalars()
    historyq = db.session.execute(db.select(History).order_by(History.id)).scalars()
        
    return render_template("user/list.html", users = users, hist = historyq, title = title)
    
        

@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        user = User.query.filter_by(email = request.form['email']).first()
        
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            flash('Login Success', 'success')
            session['user_id'] = user.id
            session['fullname'] = user.fullname
            return redirect(url_for('beranda'))
        else :
            flash('Please check your login details and try again.', 'danger')
            return render_template('user/index.html')
    
    if request.method == 'GET':
        title = "K-Genz | Login"
        return render_template("user/index.html", title = title)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(
            fullname = request.form['fullName'],
            email = request.form['email'],
            password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8'),
        )
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        return redirect(url_for('index'))
    
    
    title = "K-Genz | Register"
    return render_template("user/register.html", title = title)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/delete/<int:id>', methods=['POST', 'DELETE'])
def delete(id):
    user = db.session.execute(db.select(User).filter_by(id=id)).scalar_one()
    db.session.delete(user)
    db.session.commit()
    return 'success', 200

@app.route("/beranda")
def beranda():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        
        title = "K-Genz | Beranda"
        view = "beranda"
        return render_template("user/beranda.html", active = view, title = title, username = session['fullname'])
    
@app.route("/artikel")
def artikel(): 
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        title = "K-Genz | Artikel"
        view = "artikel"
        
        # get_artikel = getData_yt()
        get_artikel = json.load(open(os.path.join(os.path.dirname(__file__), "api/data_youtube.json")))
        
        return render_template("user/artikel.html", active = view, title = title, artikel = get_artikel["items"])
@app.route("/layanan", methods=['GET', 'POST'])
def layanan():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        title = "K-Genz | Layanan"
        view = "layanan"
        predict = 0
        
        if request.method == 'POST':
            word = request.form['review']
            predict = predict_sentiment(word)
            
            review = Review(
                review = word,
                score = int(predict),
                user_id = session['user_id']
                
            )
            
            db.session.add(review)
            db.session.commit()
            
            getReview = Review.query.order_by(Review.date.desc()).limit(5).all()
            
            
            return render_template("user/layanan.html", predict = predict, active = view, title = title, review = getReview)
            
        getReview = Review.query.order_by(Review.date.desc()).limit(5).all()
        return render_template("user/layanan.html", active = view, title = title,predict = predict, review = getReview)


@app.route("/capture-layanan")
def camera():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        title = "K-Genz | Deteksi Stress"
        return render_template("user/viewCaptureCamera.html", title = title)



@app.route("/chatbot")
def chatbot():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        title = "K-Genz | Chatbot"    
        return render_template("user/chatbot.html", title = title)

@app.route("/chatbot_res")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/history")
def history():
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        history = db.session.query(History.result, func.count(History.result)).where(History.user_id == session['user_id']).group_by(History.result).all()
        
        for history in history:
            if history[0] == 3:
                t0 = history[1]
            elif history[0] == 4:
                t1 = history[1]
            elif history[0] == 1 or history[0] == 5:
                t2 = history[1]
            elif history[0] == 2 or history[0] == 0:
                t3 = history[1]
            else:
                none = history
        
        hist = [t0, t1, t2, t3]    
        # print(session['user_id'])
        # print(history)
        print(hist)
        title = "K-Genz | History"
        return render_template("user/history.html", hist = hist, title = title)

@app.route("/profil", methods=['GET', 'POST'])
def profil():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        profil = User.query.get_or_404(session['user_id'])
        title = "K-Genz | Profil"
        view = "profil"
        
        if request.method == 'POST':
            profil.fullname = request.form['fullname']
            profil.email = request.form['email']
            db.session.commit()
            return redirect(url_for('profil'))
        
        return render_template("user/profil.html", active = view, title = title, user = profil)

@app.route("/profil_password", methods=['POST', 'GET'])    
def profil_password():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else:
        profil = User.query.get_or_404(session['user_id'])
        title = "K-Genz | Profil"
        view = "profil"
        
        if request.method == 'POST':
            pass_new = bcrypt.generate_password_hash(request.form['password_new']).decode('utf-8')
            pass_old = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            if bcrypt.check_password_hash(profil.password, request.form['password']):
                if pass_new == profil.password:
                    flash('Password baru tidak boleh sama dengan password lama', 'danger')
                    ExternalInterface.call("document.location.reload", 0);
                    return redirect(url_for('profil'))
                elif request.form['password_new'] != request.form['password_confirm']:
                    flash('Password baru tidak sesuai', 'danger')
                    return redirect(url_for('profil'))
                else:    
                    profil.password = pass_new
                    db.session.commit()
                    flash('Password berhasil diubah', 'success')
                    return redirect(url_for('profil'))
                
            else : 
                flash('Password lama tidak sesuai', 'danger')
                return redirect(url_for('profil'))
            
        return render_template("user/profil.html", active = view, title = title, user = profil)

@app.route('/uploadfile', methods=['POST'])
def upload_file():
    if session.get('user_id') is None:
        return redirect(url_for('index'))
    else :
        try:
            if request.method == 'POST':
                #usr
                user = User.query.get_or_404(session['user_id'])
                
                image = request.files['file']
                if image.filename != '':
                    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                    sh_img = image.filename
                    
                facecrop(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                img = preprocess_img(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                pred = predict_result(img)
                
                #save database history
                hist = History(
                    result = str(pred),
                    user_id = user.id
                )
                db.session.add(hist)
                db.session.commit()
                
                if pred == 4 :
                    preds = "Stress Tingkat Rendah"
                elif pred == 5 or pred == 1:
                    preds = "Stress Tingkat Sedang"
                elif pred == 2 or pred == 0:
                    preds = "Stress Tingkat Tinggi"
                else:
                    preds = "Tidak Ada Stress"

                datajson = json.load(open(os.path.join(os.path.dirname(__file__), "api/data_tingkatStress.json")))
                
                return render_template("user/resultCapture.html", sh_img=sh_img, predict=str(preds), data=datajson)
        except Exception as e:
            error = f"An error occurred: {str(e)}"
            return render_template("user/viewCaptureCamera.html", message=error)
    
    
@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    if data and 'image' in data:
        img_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(img_data)))
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], "capture-camera.png"))
        img = preprocess_img(io.BytesIO(base64.b64decode(img_data)))
        pred = predict_result(img)
        return 'success', 200
    return 'Error', 400

@app.route('/predict')
def predict():
    facecrop(os.path.join(app.config['UPLOAD_FOLDER'], "capture-camera.png"))
    img = preprocess_img(os.path.join(app.config['UPLOAD_FOLDER'], "capture-camera.png"))
    pred = predict_result(img)
    
    hist = History(
        result = str(pred),
        user_id = session.get('user_id')
    )
    db.session.add(hist)
    db.session.commit()
    
    if pred == 4 :
        preds = "Stress Tingkat Rendah"
    elif pred == 5 or pred == 1:
        preds = "Stress Tingkat Sedang"
    elif pred == 2 or pred == 0:
        preds = "Stress Tingkat Tinggi"
    else:
        preds = "Tidak Ada Stress"
        
    datajson = json.load(open(os.path.join(os.path.dirname(__file__), "api/data_tingkatStress.json")))
        
    return render_template("user/resultCapture.html", sh_img="capture-camera.png", predict=str(preds), data = datajson)
    
if __name__ == '__main__':
	app.run(debug=True)
