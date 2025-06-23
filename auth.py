import pyrebase


firebaseConfig = {
    'apiKey': "AIzaSyD9gHw7bBJNoeuBe71w5k2cBSOVS76qnp0",
    'authDomain': "nbafantasy-d9051.firebaseapp.com",
    'projectId': "nbafantasy-d9051",
    'storageBucket': "nbafantasy-d9051.firebasestorage.app",
    'messagingSenderId': "755210883929",
    'appId': "1:755210883929:web:0ecc61d24fa87882cef248",
    'measurementId': "G-81C87ZFBH5",
    'databaseURL': ''
}

auth = None

def firebaseInit():
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()
    return auth

def signInEmail(email, password):
    user = auth.sign_in_with_email_and_password(email, password)

def signUpEmail(email, password):
    user = auth.create_user_with_email_and_password(email, password)
    auth.send_email_verification(user['idToken'])

def resetPassword(email):
    auth.send_password_reset_email(email)