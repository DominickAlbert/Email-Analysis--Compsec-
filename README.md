# Phishing Email Detection System

## Overview

This project implements a **Phishing Email Detection System** using machine learning techniques. The system analyzes email metadata (e.g., sender, receiver, date, subject, body, URLs) to classify emails as legitimate or phishing.

The project integrates with the **Gmail API** to notify users about high-risk emails in real-time, ensuring timely alerts for potential threats.

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv). It contains labeled emails with features such as sender, receiver, subject, body, and URLs, making it suitable for training and evaluating phishing detection models.

# Phishing Email Seriousness Scoring System

This document explains the scoring system used to evaluate the seriousness level of phishing emails. Each component is assigned a score based on its likelihood of indicating phishing activity.

---

## 1. URL Presence (`urls`) - Score Contribution: +0.35

**Explanation**:  
URLs are a primary method for phishing attacks. Attackers often embed malicious links in emails to steal credentials or deliver malware.

**Rationale**:  
Including a URL significantly increases the risk of phishing, so it gets a high weight (35%). However, not all emails with URLs are malicious, so other factors are still considered.

---

## 2. Sender Domain Check (`sender_domain`) - Score Contribution: +0.25

**Explanation**:  
Phishing emails often come from untrusted or spoofed domains. Trusted domains like Gmail or Outlook are less likely to be used in phishing campaigns.

**Rationale**:  
If the sender domain is not in the trusted list, it raises suspicion. Assigning 25% ensures this is a significant but not overwhelming factor, as legitimate emails can also come from unknown domains.

---

## 3. Keyword Analysis (`clean_subject` and `clean_body`) - Score Contribution: Up to +0.30

**Explanation**:  
Phishing emails frequently use specific keywords to create urgency or fear, such as "urgent," "verify," or "account suspended." These words are designed to pressure recipients into taking immediate action.

**Rationale**:  
Each keyword match contributes a small score (e.g., 7% per keyword), with a cap at 30%. This prevents over-penalizing emails with many matches while still flagging suspicious content.

---

## 4. Timing Analysis (`day_of_week` and `hour_normalized`) - Score Contribution: +0.10 to +0.15

**Explanation**:  
Phishing emails are often sent during unusual times (e.g., weekends or late at night) when recipients may be less vigilant.

**Rationale**:  
- Emails sent on weekends (`day_of_week = 6 or 7`) get +0.10 because attackers may target users when they're less likely to verify the email's legitimacy.
- Emails sent during unusual hours (before 6 AM or after 8 PM) get +0.15 because these times are outside typical business hours, increasing suspicion.

---

## 5. Domain Mismatch Check (`sender_domain` vs. `receiver_domain`) - Score Contribution: +0.15

**Explanation**:  
Legitimate internal communications usually occur between matching domains (e.g., `company.com` to `company.com`). A mismatch suggests the email originated externally.

**Rationale**:  
Assigning 15% ensures this is a meaningful factor without overshadowing other indicators, as cross-domain emails can still be legitimate.

---

## 6. Suspicious Sender Domain Patterns (`sender_domain`) - Score Contribution: +0.10

**Explanation**:  
Phishers often use deceptive domain names, such as adding numbers or hyphens to mimic legitimate domains (e.g., `gma1l.com` instead of `gmail.com`).

**Rationale**:  
Domains with suspicious patterns (numbers, hyphens, etc.) are more likely to be malicious. Adding 10% penalizes these patterns while keeping the score balanced.

---

## Final Score Normalization

The total score is capped at **1.0** to ensure consistency. This allows for easy interpretation:
- **0.0 - 0.3**: Low risk (likely legitimate)
- **0.3 - 0.6**: Moderate risk (requires further review)
- **0.6 - 1.0**: High risk (likely phishing)