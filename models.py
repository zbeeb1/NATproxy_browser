from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()
db = SQLAlchemy()

class CachedResponse(db.Model):
    __tablename__ = 'cached_response'
    id = db.Column(db.Integer, primary_key=True)
    method = db.Column(db.String(10))
    url = db.Column(db.String(255))
    params = db.Column(db.String(255))
    headers = db.Column(db.String(255))
    content = db.Column(db.LargeBinary)
    timestamp = db.Column(db.DateTime)
    last_modified = db.Column(db.String(50))
    protocol = db.Column(db.String(5))  
    user_id = db.Column(db.Integer)       

    def associate_with_user(self, user_id):
        self.user_id = user_id
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    blocked_urls = db.Column(db.String(255))
    is_blocked = db.Column(db.Boolean, default=False)  # Add the 'is_blocked' attribute  
    
    def get_blocked_urls(self):
        if self.blocked_urls:
            return self.blocked_urls.split(',')
        return []  
    def set_blocked_urls(self, urls_list):
        self.blocked_urls = ','.join(urls_list)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)