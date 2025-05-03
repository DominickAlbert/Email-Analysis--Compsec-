import os
import re
import base64
from email.mime.text import MIMEText
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Your existing imports and functions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",   # read + mark-as-read
    "https://www.googleapis.com/auth/gmail.send"     # send messages
]

ADMIN_EMAIL = "server2004phising@gmail.com"
PHISHING_THRESHOLD = 0.5

stop_words = set(stopwords.words("english"))

def authenticate_gmail():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def list_unread_messages(service):
    resp = service.users().messages().list(
        userId="me",
        labelIds=["INBOX", "UNREAD"],
        maxResults=20
    ).execute()
    return resp.get("messages", [])

def get_message_payload(service, msg_id):
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = { h["name"]: h["value"] for h in msg["payload"]["headers"] }
    subject = headers.get("Subject", "")
    sender = headers.get("From", "")
    to = headers.get("To", "")
    date_str = headers.get("Date", "")
    # parse body (naïve—handle text/plain at top level or first part)
    parts = msg["payload"].get("parts", [])
    body = ""
    for part in parts:
        if part.get("mimeType") == "text/plain":
            data = part["body"]["data"]
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            break
    return subject, body, sender, to, date_str, msg

def mark_as_read(service, msg):
    service.users().messages().modify(
        userId="me",
        id=msg["id"],
        body={ "removeLabelIds": ["UNREAD"] }
    ).execute()

def create_message(to, subject, body_text):
    msg = MIMEText(body_text)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return { "raw": raw }

def send_message(service, message):
    return service.users().messages().send(userId="me", body=message).execute()

# Re-use your cleaning & scoring functions:
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text)
    tokens = word_tokenize(text)
    return " ".join(w.lower() for w in tokens if w.lower() not in stop_words)

def calculate_phishing_seriousness(clean_subject, clean_body, sender_domain,
        receiver_domain, day_of_week, hour_normalized, urls):
    score = 0.0
    if urls == 1: score += 0.35
    trusted_domains = {'gmail.com', 'outlook.com', 'yahoo.com', 'hotmail.com', 'protonmail.com',
    'icloud.com', 'aol.com', 'zoho.com', 'gmx.com', 'mail.com', 'tutanota.com',
    'fastmail.com', 'hushmail.com', 'runbox.com', 'posteo.de', 'disroot.org'
    }
    if sender_domain not in trusted_domains:
        score += 0.25
    # keyword hits
    phishing_keywords = {
    'password', 'urgent', 'verify', 'account', 'login', 'bank', 'security', 
    'suspended', 'confirm', 'fraud', 'update', 'alert', 'compromised', 
    'immediately', 'limited', 'action', 'required', 'personal', 'information',
    'click', 'link', 'attachment', 'unauthorized', 'activity', 'locked', 
    'expired', 'reactivate', 'invoice', 'payment', 'refund', 'transaction'
    }
    text = (clean_subject + ' ' + clean_body).lower()
    keyword_hits = sum(1 for word in phishing_keywords if word in text)
    score += min(keyword_hits * 0.07, 0.6) 
    # timing
    if day_of_week in {6,7}: score += 0.1
    if hour_normalized < 6/24 or hour_normalized > 20/24: score += 0.15
    if sender_domain.split("@")[-1] != receiver_domain.split("@")[-1]: score += 0.15
    if any(c in sender_domain for c in "-0123"): score += 0.1
    return max(0.0, min(score,1.0))

def main():
    service = authenticate_gmail()

    import pickle
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))

    unread = list_unread_messages(service)
    print(len(unread))
    for m in unread:
        print("tess new email")
        subj, body, frm, to, date_str, msg_obj = get_message_payload(service, m["id"])
        # preprocess fields:
        clean_subj = clean_text(subj)
        clean_body = clean_text(body)
        sender_dom = frm.split("@")[-1].strip(">")
        recv_dom = to.split("@")[-1].strip(">")
        dt = pd.to_datetime(date_str, errors="coerce")
        hour_norm = dt.hour / 23.0
        dow = dt.dayofweek
        has_url = int(bool(re.search(r"https?://", body)))

        # ML prediction
        combined = clean_subj + " " + clean_body
        x = vectorizer.transform([combined])
        prob_phish = model.predict_proba(x)[0][1]

        # rule-based score
        score_rb = calculate_phishing_seriousness(
            clean_subj, clean_body, sender_dom, recv_dom, dow, hour_norm, has_url
        )

        threat = prob_phish*score_rb

        print("threat level : ", threat)
        if threat >= PHISHING_THRESHOLD:
            text = (
                f"⚠️ *Potential Phishing Alert*\n\n"
                f"From: {frm}\nTo: {to}\nSubject: {subj}\n"
                f"Detected phishing probability: {prob_phish:.2f}\n"
                f"Rule-based score: {score_rb:.2f}\n"
                f"Final threat level: {threat:.2f}\n\n"
                f"---\n\n{body[:500]}..."
            )
            msg = create_message(ADMIN_EMAIL, f"[ALERT] Phishing ({threat:.2f})", text)
            send_message(service, msg)

        # mark as read so we don’t re-process
        mark_as_read(service, msg_obj)

if __name__ == "__main__":
    main()
