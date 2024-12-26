from models import db


class UserFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(80), db.ForeignKey(
        'user.username'), nullable=False)
    status = db.Column(db.Boolean, default=False)
    feedback = db.Column(db.Text, nullable=False)
