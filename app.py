#inport necessary libraries 
from flask import Flask, render_template, request, Response, jsonify,session
from flask_sqlalchemy import SQLAlchemy
import requests
from datetime import datetime
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier   
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address  
from urllib.parse import urlparse
import numpy as np
import socket
import random
import re  
import uuid  
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for

from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_bcrypt import Bcrypt
from models import db, User, CachedResponse
from flask_login import LoginManager, login_user, login_required, logout_user, current_user   
from flask_login import logout_user

app = Flask(__name__)#intialize flask
logger = logging.getLogger('logs.txt')# intialize logger file
handler = logging.handlers.RotatingFileHandler('logs.txt', mode='w', maxBytes=10000, backupCount=1)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.info("hello world")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cache.db'#config data base
app.config['SECRET_KEY'] = 'zbeebzbeeb' #config key 
#intialize necessay flask libraries(hashing login for passwords)
with app.app_context():
    db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
#device information
mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
print("   @current working mac address lenovo ideapad:", mac)
logger.info(f"@current working mac address lenovo ideapad: {mac}")

#ddos attack prevention(using flask limiter)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)
print("you are now protected from DDOS attacks thanks to python3 flask limiter tool")
logger.info("you are now protected from DDOS attacks thanks to python3 flask limiter tool")
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print("local IP Address:", ip_address)  
logger.info(f"local IP Address: {ip_address}")
print("creating database sqllite") 
logger.info("creating database sqllite")
#cutomizing of directing handeling
class RedirectHandlingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    from_url = db.Column(db.String(512))
    to_url = db.Column(db.String(512))
    label = db.Column(db.String(10))

version = '1.2.1-dkxce'
module = 'flask-proxy'
allow_redirect = True

 

#setting up  machine learning libraries and training regression model and random forest clasifire data set 
cache_expiration_model = LinearRegression()
redirect_handling_model = RandomForestClassifier()

elapsed_times = [random.uniform(10000, 500000) for _ in range(10)]
cache_times = [random.randint(200, 2000) for _ in range(10)]
#setting up ML graph 
X_train = np.array([[time] for time in elapsed_times])  
print("training the X-")
logger.info("training the X-")
y_train = np.array(cache_times)
print("training the Y-") 
logger.info("training the Y-")
cache_expiration_model.fit(X_train, y_train)
print("fiting the data")
logger.info("fiting the data")
CACHE_LIMIT = 1000  #cache limit

#order dictionary for Least Recently Used
cache_queue = OrderedDict() 
print("Cache Expiration Model Coefficients:", cache_expiration_model.coef_)
print("Cache Expiration Model Intercept:", cache_expiration_model.intercept_)  
logger.info(f"Cache Expiration Model Coefficients: {cache_expiration_model.coef_}")
logger.info(f"Cache Expiration Model Intercept: {cache_expiration_model.intercept_}")
#fitting the graph

predictions_train = cache_expiration_model.predict(X_train) 

prediction = cache_expiration_model.predict(X_train)[0] 
CACHE_EXPIRATION_TIME = max(prediction ,100000)#setting cache expiration time based on ML 
print("Predictions on Training Data:")
logger.info("Predictions on Training Data:")
for i, pred in enumerate(predictions_train):
    print(f"Actual: {y_train[i]}, Predicted: {pred}")
    logger.info((f"Actual: {y_train[i]}, Predicted: {pred}"))

def proxy_view(url, requests_args=None, *args, **kwargs):#core funtions args kwargs for overriding
    try:
        #parsing and handeling requests
        requests_args = requests_args or {}
        params = request.args

    
        # Minimized header manipulation
        headers = {key: value for key, value in request.headers.items() if key.lower() not in ['content-length']}
        
        requests_args['headers'] = headers
        requests_args['params'] = params
        user_id = session.get('user_id')
        
        # Determine the URL protocol (HTTP or HTTPS)
        protocol = url.split('://')[0].lower()

        # Create a cache key with protocol information
        cache_key = f'{request.method}_{protocol}_{url}'

        # Check if the URL protocol is HTTPS and user is authenticated
        user_id = session.get('user_id')
        if protocol == 'https' and user_id is not None:
            # Append user-specific information to the HTTPS cache key
            cache_key = f'user_{user_id}_{cache_key}'

        # Attempt to retrieve from cache
        cached_response = CachedResponse.query.filter_by(
            method=request.method,
            url=url,
            protocol=protocol  # Add protocol filter
        ).first()

        if cached_response:
            print("Checking if modified since and the expiration of the cache")
            logger.info("Checking if modified since and the expiration of the cache")
            if (datetime.utcnow() - cached_response.timestamp).total_seconds() < CACHE_EXPIRATION_TIME:
                print("Returning cached response correctly")
                logger.info("Returning cached response correctly")
                return Response(cached_response.content, status=200, headers={'Content-Type': 'cached'})
            else:
                print("Cached response expired")
                logger.info("Cached response expired")
        else:
            print("No cached response found")
            logger.info("No cached response found")
        # Fetching URL from the internet
        print("Fetching URL from the internet") 
        logger.info("Fetching URL from the internet")
        start_time = datetime.utcnow() 
        response = requests.request(request.method, url, **requests_args)
        end_time = datetime.utcnow()
        print(f"Response received from the internet at {end_time}") 
        logger.info(f"Response received from the internet at {end_time}")
        print("Creating the cache")
        logger.info("Creating the cache")
        new_cached_response = CachedResponse(
            method=request.method,
            url=url,
            content=response.content,
            timestamp=datetime.utcnow(),
            last_modified=response.headers.get('Last-Modified', ''),
            protocol=protocol  # Store the protocol in the cache
        )  
        db.session.add(new_cached_response)
        try:
            db.session.commit()
            print("New response cached and saved well in the database")
            logger.info("New response cached and saved well in the database")

        except Exception as e:
            print(f"Error during database commit: {e}")
            logger.info(f"Error during database commit: {e}")
        proxy_response = Response(response.content, status=response.status_code)
        #excluding headers for advanced storage
        excluded_headers = set([
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding',
            'upgrade', 'content-encoding', 'content-length',
        ])
        for key, value in response.headers.items():
            if key.lower() in excluded_headers:
                continue
            else:
                proxy_response.headers[key] = value

        uri = urlparse(url)
        try:
            HOSTNAME = socket.gethostname()
        except:
            HOSTNAME = 'localhost'
        proxy_response.headers["Via"] = f"Proxy Server; host={HOSTNAME}; proto={uri.scheme}"
        proxy_response.headers["Forwarded"] = f"for={request.remote_addr}; host={HOSTNAME}; proto={uri.scheme}"
        print("Response sent to the client")
        logger.info("Response sent to the client")
        print(f"Time received from the internet: {start_time}")
        logging.info(f"Time received from the internet: {start_time}")
        return proxy_response

    except requests.RequestException as e:
        print(f"Error: {e}")
        return Response("Error occurred", status=500)


#detecting traffic based on ML (reffer to the report to check how the ml tool works)
def detect_traffic_anomaly(data):
    features = np.array([[data['elapsed_time']]])
    prediction = cache_expiration_model.predict(features)
    return prediction < 0
#update cache based on LRU 
def update_cache_access(cached_item):
    global cache_queue
    cache_queue.pop(cached_item.id, None)
    cache_queue[cached_item.id] = cached_item
    if len(cache_queue) > CACHE_LIMIT:
        oldest_item_id = next(iter(cache_queue))
        print(f"Evicting cache item with ID: {oldest_item_id}")
        db.session.delete(CachedResponse.query.get(oldest_item_id))
        db.session.commit()
        cache_queue.popitem(last=False)

#parsing additional information from url
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path_length = len(parsed_url.path)
    return [domain, path_length]
#redirecting malicious redirections 
def log_alert(message):
    print(f"ALERT: {message}")  

def classify_and_alert_redirect(url):
    features = extract_features(url)
    prediction = redirect_handling_model.predict([features])
    if prediction == 1:  
        log_alert(f"Potentially malicious redirect detected: {url}")
    return prediction

def make_absolute_location(base_url, location):
    absolute_pattern = re.compile(r'^[a-zA-Z]+://.*$')
    if absolute_pattern.match(location):
        return location

    parsed_url = urlparse(base_url)
    if location.startswith('//'):
        return parsed_url.scheme + ':' + location
    elif location.startswith('/'):
        return parsed_url.scheme + '://' + parsed_url.netloc + location
    else:
        return parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path.rsplit('/', 1)[0] + '/' + location
#MAIN flask function
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    try:
        if current_user.is_authenticated and current_user.email == "natproxyy@outlook.com":
            return render_template('admin.html')

        if request.method == 'POST':
            url = request.form.get('url')
            if url:
                # Check if the URL is in the blocked_urls list for the current user
                blocked_urls = current_user.get_blocked_urls()
                if url in blocked_urls :
                    return jsonify({'error': 'invaid input'})

                start_time = datetime.utcnow() 
                response = proxy_view(url, None)
                elapsed_time = (datetime.utcnow() - start_time).total_seconds()
                anomaly_data = {'elapsed_time': elapsed_time}

                if detect_traffic_anomaly(anomaly_data):
                    logger.info("Traffic anomaly detected!")

                return jsonify({
                    'requested_content': response.get_data(as_text=True),
                    'requested_url': url
                })

        return render_template('index.html')
    except Exception as e:
        # Handle exceptions here
        return jsonify({'error': str(e)})

#tested and passed by zbeeb 
@app.route('/admin/block_user', methods=['POST'])
@login_required
def block_user(): 
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    user_email = request.form.get('email')
    user = User.query.filter_by(email=user_email).first()

    if user:
        user.is_blocked = True  # Set the 'is_blocked' attribute to True
        db.session.commit()
        return jsonify({"message": f"User {user_email} blocked"}), 200
    else:
        return jsonify({"error": "User not found"}), 404   

#tested and passed by zbeeb
@app.route('/admin/block_url_for_all', methods=['POST'])
@login_required
def block_url_for_all():
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    url = request.form.get('url')

    # Validate input
    if not url:
        return jsonify({"error": "Invalid URL"}), 400

    users = User.query.all()
    try:
        for user in users: 
            if user : 
                blocked_urls = user.get_blocked_urls()  # Get the list of blocked URLs
                blocked_urls.append(url)  # Append the URL to the list
                user.set_blocked_urls(blocked_urls)  # Set the modified list back to the user
        db.session.commit()
        return jsonify({"message": f"URL {url} blocked for all users"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

#tested and passed Mariam 
@app.route('/admin/block_url_for_user', methods=['POST'])
@login_required
def block_url_for_user():
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    user_email = request.form.get('email')
    url = request.form.get('url')
    user = User.query.filter_by(email=user_email).first()
    
    if user:
        blocked_urls = user.get_blocked_urls()  # Get the list of blocked URLs
        blocked_urls.append(url)  # Append the URL to the list
        user.set_blocked_urls(blocked_urls)  # Set the modified list back to the user
        db.session.commit()  # Commit the changes to the database 
        print("url blocked correctly")
        return jsonify({"message": "URL blocked successfully"})
    else: 
        print("user not found")
        return jsonify({"error": "User not found"}), 404

#tested and passed By bazzy (everyone dont touch the java script please)
@app.route('/admin/allow_user_access', methods=['POST'])
@login_required
def allow_user_access(): 
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    user_email = request.form.get('email')
    user = User.query.filter_by(email=user_email).first()

    if user:
        user.is_blocked = False  # Set the 'is_blocked' attribute to False
        db.session.commit()
        return jsonify({"message": f"User {user_email} access allowed"}), 200
    else:
        return jsonify({"error": "User not found"}), 404

#tested and passed by ghorayeb
@app.route('/admin/unblock_url_for_user', methods=['POST'])
@login_required
def unblock_url_for_user():
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    user_email = request.form.get('email')
    url = request.form.get('url')
    user = User.query.filter_by(email=user_email).first()
    if user: 
        mylist = user.get_blocked_urls()  
        for urls in mylist  : 

                if urls == url :  
                    mylist.remove(urls)     
                    break   
        user.set_blocked_urls(mylist)
        db.session.commit()
        return jsonify({"message": f"URL {url} unblocked for user {user_email}"}), 200
    else:
        return jsonify({"error": "User not found"}), 404


    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    user_email = request.form.get('email')
    url = request.form.get('url')
    user = User.query.filter_by(email=user_email).first()

    if user:
        blocked_urls = user.get_blocked_urls()  # Get the list of blocked URLs
        if url in blocked_urls:
            blocked_urls.remove(url)  # Remove the URL from the list
            user.set_blocked_urls(blocked_urls)  # Set the modified list back to the user
            db.session.commit()
            return jsonify({"message": f"URL {url} unblocked for user {user_email}"}), 200
        else:
            return jsonify({"error": "URL not found in user's blocked list"}), 400
    else:
        return jsonify({"error": "User not found"}), 404


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login(): 
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
       
        if user and user.check_password(password):
            if user.is_blocked:
                
                return redirect(url_for('login'))
            login_user(user)
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            
            sender_email = "natproxyy@outlook.com"
            sender_password ="NATPROXY123"
            receiver_email = email
            smtp_server = "smtp-mail.outlook.com"
            smtp_port = 587

            # Create a MIME object
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = "Welcome to GoProxy – Your Campus Internet Security Solution!"

            #Verification email content
           
            body = ("Welcome to NATproxyy!🎉\n" 
            "We're thrilled to have you join our community dedicated to ensuring safe and secure browsing experiences, "
            "while harnessing the power of machine learning to personalize and enhance your online journey.\n"
            "At NATproxyy, our primary goal is to provide a robust and reliable platform that safeguards your browsing activities while leveraging "
            "the latest advancements in machine learning technology.\n"
            "We understand the significance of privacy and security, especially in today's digital landscape, and our team is committed to delivering "
            "a seamless and protected online environment for each of our users.")


            message.attach(MIMEText(body, "plain"))

            # Establish a connection to the SMTP server
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Use TLS for security
                server.login(sender_email, sender_password)

                # Send the email
                server.sendmail(sender_email, receiver_email, message.as_string())

            print("Welcome email sent successfully!")
            logger.info("Welcome email sent successfully!")
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
  
#tested and passed
@app.route('/admin/unblock_all_users', methods=['POST'])
@login_required
def unblock_all_user():    
    if current_user.email != "natproxyy@outlook.com":
        return jsonify({"error": "Unauthorized"}), 403

    url = request.form.get('url')
    users = User.query.all()    
    for u in users:  
        if u:
            blocked_urls = u.get_blocked_urls()
            if url in blocked_urls:
                blocked_urls.remove(url)
                u.set_blocked_urls(blocked_urls)
    db.session.commit()  # This should be outside the loop
    return jsonify({"message": f"URL {url} unblocked for all users"}), 200

def create_users():
    # Create the User table if it doesn't exist
    db.session.execute(
        text("CREATE TABLE IF NOT EXISTS user (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, blocked_urls TEXT, is_blocked BOOLEAN DEFAULT 0)")
    )

    #the data base of added users
    if not User.query.all():
        user1 = User(email='mbz02@mail.aub.edu' ,   is_blocked=False) 
        user1.set_blocked_urls(["https://leetcode.com/"])
        user1.set_password('12345')
        db.session.add(user1)

        user2 = User(email='mrg11@mail.aub.edu', is_blocked=False)
        user2.set_password('12345')
        db.session.add(user2)

        user3 = User(email='mcs12@mail.aub.edu', is_blocked=False)
        user3.set_password('12345')
        db.session.add(user3)

        user4 = User(email='mab101@mail.aub.edu',  is_blocked=False)
        user4.set_password('12345')
        db.session.add(user4)

        user5 = User(email="natproxyy@outlook.com",  is_blocked=False)
        user5.set_password('NATPROXY123')   
        db.session.add(user5)

        user6 = User(email="at56@mail.aub.edu", is_blocked=False) #for testing (ayman tajjed)
        user6.set_password('12345')
        db.session.add(user6)

    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_users()

    app.run(port=5000, debug=True)