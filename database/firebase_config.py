import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

load_dotenv()


def init_firebase():
    key_path = os.getenv("FIREBASE_KEY_PATH")
    db_url = os.getenv("FIREBASE_DB_URL")
    db_bucket = os.getenv("FIREBASE_BUCKET_URL")

    cred = credentials.Certificate(key_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": db_url,
             "storageBucket": db_bucket},
        )
    return db


def check_connection():
    try:
        test_ref = db.reference("health_check")
        health_check = test_ref.get()
        if health_check == {"ping": "ok"}:
            return {
                "status": "success",
                "message": "Connected to Firebase",
            }
        else:
            return {
                "status": "failed",
                "message": "Invalid or missing health check data",
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
