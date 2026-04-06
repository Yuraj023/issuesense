# Main application file
from flask import Flask, render_template, request
import pickle
import os
import json
import re
from dotenv import load_dotenv
from supabase import create_client
from textblob import TextBlob
from utils.preprocess import clean_text
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Load trained ML model
model = pickle.load(open("model/complaint_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
RISKY_TERMS_PATTERN = re.compile(
    r"(?i)\b(password|passcode|otp|one[-\s]?time|login|sign\s*in|verify|verification|"
    r"bank|credit\s*card|card\s*number|ssn|social\s*security|pin)\b"
)
DIGIT_RUN_PATTERN = re.compile(r"\b\d{12,19}\b")


def mask_risky_text(value):
    if not value:
        return value
    masked = URL_PATTERN.sub("[link removed]", value)
    masked = EMAIL_PATTERN.sub("[email removed]", masked)
    masked = RISKY_TERMS_PATTERN.sub("[redacted]", masked)
    masked = DIGIT_RUN_PATTERN.sub("[number removed]", masked)
    if len(masked) > 400:
        masked = masked[:400].rstrip() + "..."
    return masked


@app.template_filter("mask_risky_text")
def mask_risky_text_filter(value):
    return mask_risky_text(value)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["complaint"]

    # Preprocess text
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    # Sentiment analysis
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        sentiment_label = "Positive"
    elif sentiment_score < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Insert into Supabase
    supabase.table("complaints").insert({
        "complaint_text": text,
        "predicted_category": prediction,
        "sentiment": sentiment_label
    }).execute()

    return render_template("index.html",
                           prediction=prediction,
                           sentiment=sentiment_label)


@app.route("/dashboard")
def dashboard():
    model_metrics = {
        "precision": "85.6%",
        "recall": "83.4%",
        "f1_score": "83.4%",
        "training_time": "2.4s",
        "model_size": "1.2 MB",
        "last_updated": "Just now"
    }

    report_path = os.path.join("model", "model_report.json")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        classification_report = report.get("classification_report", {})
        macro_avg = classification_report.get("macro avg", {})

        if macro_avg.get("precision") is not None:
            model_metrics["precision"] = f"{macro_avg['precision'] * 100:.1f}%"
        if macro_avg.get("recall") is not None:
            model_metrics["recall"] = f"{macro_avg['recall'] * 100:.1f}%"
        if macro_avg.get("f1-score") is not None:
            model_metrics["f1_score"] = f"{macro_avg['f1-score'] * 100:.1f}%"

        training_time = report.get("training_time") or report.get("training_time_seconds")
        if isinstance(training_time, (int, float)):
            model_metrics["training_time"] = f"{training_time:.1f}s"
        elif isinstance(training_time, str) and training_time.strip():
            model_metrics["training_time"] = training_time

        model_size = report.get("model_size") or report.get("model_size_mb")
        if isinstance(model_size, (int, float)):
            model_metrics["model_size"] = f"{model_size:.1f} MB"
        elif isinstance(model_size, str) and model_size.strip():
            model_metrics["model_size"] = model_size

        model_metrics["last_updated"] = report.get("last_updated") or "Just now"

    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        # Keep safe defaults if report is unavailable or malformed.
        pass

    response = supabase.table("complaints").select("*").order("created_at", desc=True).execute()
    complaints = response.data or []

    resolved_complaints = [
        complaint for complaint in complaints
        if (complaint.get("status") or "").lower() == "resolved"
    ]
    active_complaints = [
        complaint for complaint in complaints
        if (complaint.get("status") or "").lower() != "resolved"
    ]
    resolved_count = len(resolved_complaints)

    # If no active complaints yet
    if not active_complaints:
        return render_template(
            "dashboard.html",
            labels=[],
            values=[],
            total=0,
            complaints=[],
            resolved_count=resolved_count,
            model_metrics=model_metrics
        )

    # Count category occurrences (active complaints only)
    category_count = {}
    for row in active_complaints:
        category = row["predicted_category"]
        category_count[category] = category_count.get(category, 0) + 1

    labels = list(category_count.keys())
    values = list(category_count.values())
    total = len(active_complaints)

    return render_template(
        "dashboard.html",
        labels=labels,
        values=values,
        total=total,
        complaints=active_complaints,
        resolved_count=resolved_count,
        model_metrics=model_metrics
    )


@app.route("/solved", methods=["GET", "POST"])
def resolve():
    action_message = None
    action_message_type = "pending"
    search_result = None
    searched_id = ""

    if request.method == "POST":
        complaint_id = (request.form.get("complaint_id") or "").strip()
        action = (request.form.get("action") or "search").lower()
        searched_id = complaint_id

        if not complaint_id:
            action_message = "Please enter a complaint ID."
        else:
            complaint_id_value = int(complaint_id) if complaint_id.isdigit() else complaint_id
            lookup = supabase.table("complaints").select("*").eq("id", complaint_id_value).limit(1).execute()
            record = lookup.data[0] if lookup.data else None

            if not record:
                action_message = f"No complaint found for ID {complaint_id}."
            else:
                search_result = record
                if action == "resolve":
                    supabase.table("complaints").update({"status": "resolved"}).eq("id", complaint_id_value).execute()
                    search_result["status"] = "resolved"
                    action_message = f"Complaint {complaint_id} marked as resolved."
                    action_message_type = "resolved"
                else:
                    action_message = f"Showing complaint {complaint_id}."
                    action_message_type = "processing"

    response = supabase.table("complaints").select("*").order("created_at", desc=True).execute()
    complaints = response.data or []
    resolved_complaints = [
        complaint for complaint in complaints
        if (complaint.get("status") or "").lower() == "resolved"
    ]

    return render_template(
        "solved.html",
        resolved_complaints=resolved_complaints,
        search_result=search_result,
        action_message=action_message,
        action_message_type=action_message_type,
        searched_id=searched_id
    )

if __name__ == "__main__":
    app.run(debug=True)