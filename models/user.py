from flask_sqlalchemy import SQLAlchemy
from models import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    unique_id = db.Column(db.String(36), nullable=False, unique=True)
