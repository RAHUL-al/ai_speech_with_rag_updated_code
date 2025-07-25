import firebase_admin
from firebase_admin import credentials, firestore

# Initialize app only once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
