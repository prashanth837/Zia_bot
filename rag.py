import numpy as np
import faiss
import os
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import aiohttp
from io import BytesIO
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes

# =============================
# 🔐 LOAD ENV
# =============================
load_dotenv()

BOT_TOKEN = "YOUR_TOKEN_HERE"  # keep as is if you want
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"

INFO_SHEET_ID = "1kUvOq9_HqBVk6dlfnDpMV7FJ9GbGSGXtrrC1zB6O5Oc"
PDF_SHEET_ID = "1ME1I3OyFS9VYH2qeqHA5Elt9_f0XXNkkmDgyreVLylo"

# =============================
# 🧠 MEMORY
# =============================
USER_MEMORY = {}

# =============================
# 📊 GOOGLE SHEETS
# =============================
import json

if os.getenv("GOOGLE_CREDENTIALS_JSON"):
    creds_info = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )
else:
    creds = Credentials.from_service_account_file(
        "credentials.json",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )

client = gspread.authorize(creds)

info_sheet = client.open_by_key(INFO_SHEET_ID).sheet1
pdf_sheet = client.open_by_key(PDF_SHEET_ID).sheet1

# =============================
# LOAD DATA
# =============================
def load_data():
    records = info_sheet.get_all_records()
    texts, answers = [], []

    for row in records:
        keywords = str(row.get("keywords", ""))
        info = str(row.get("answer", ""))

        if not info.strip():
            continue

        texts.append(f"keywords: {keywords} | info: {info}")
        answers.append(info)

    return texts, answers

texts, answers = load_data()

# =============================
# EMBEDDINGS
# =============================
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embed_model.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# =============================
# PDF VECTOR DB
# =============================
pdf_texts, pdf_meta = [], []

for row in pdf_sheet.get_all_records():
    k = str(row.get("keyword", ""))
    name = str(row.get("file_name", ""))
    url = str(row.get("file_url", ""))

    if k and url:
        pdf_texts.append(k)
        pdf_meta.append((name, url))

pdf_embeddings = embed_model.encode(pdf_texts)
pdf_index = faiss.IndexFlatL2(pdf_embeddings.shape[1])
pdf_index.add(np.array(pdf_embeddings))

# =============================
# SEARCH
# =============================
def search_pdf(query):
    q = embed_model.encode([query])
    D, I = pdf_index.search(np.array(q), 1)
    if D[0][0] < 1.0:
        return pdf_meta[I[0][0]]
    return None, None

def retrieve(query):
    q = embed_model.encode([query])
    D, I = index.search(np.array(q), 2)
    return [(answers[i], score) for i, score in zip(I[0], D[0])]

# =============================
# SEND PDF
# =============================
async def send_pdf(update, name, url):
    await update.message.reply_text("📄 Sending file...")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            file_bytes = BytesIO(await resp.read())

            await update.message.reply_document(
                document=file_bytes,
                filename=name
            )

# =============================
# MAIN HANDLER
# =============================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.text:
        return

    user_id = update.message.from_user.id
    text = update.message.text

    if user_id not in USER_MEMORY:
        USER_MEMORY[user_id] = []

    USER_MEMORY[user_id].append(f"User: {text}")

    # PDF
    name, url = search_pdf(text)
    if url:
        await send_pdf(update, name, url)
        return

    # RAG
    results = retrieve(text)
    filtered = [t for t, s in results if s < 0.8]

    if filtered:
        context_text = "\n".join(filtered)

        try:
            model = genai.GenerativeModel(MODEL_NAME)

            prompt = f"""
            Answer ONLY using this information with a polished tone:

            {context_text}

            Question: {text}
            """

            res = model.generate_content(prompt)
            answer = res.text.strip()

        except:
            answer = filtered[0]

    else:
        history = "\n".join(USER_MEMORY[user_id][-5:])

        try:
            model = genai.GenerativeModel(MODEL_NAME)

            prompt = f"""
            Continue conversation.

            {history}

            User: {text}
            """

            res = model.generate_content(prompt)
            answer = res.text.strip()

        except:
            answer = "⚠ AI busy, try later."

    USER_MEMORY[user_id].append(answer)

    await update.message.reply_text(answer)

# =============================
# FLASK WEBHOOK (RAILWAY)
# =============================
from flask import Flask, request
import asyncio
from telegram import Bot, Update

app_flask = Flask(__name__)
bot = Bot(BOT_TOKEN)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@app_flask.route("/", methods=["POST"])
def webhook():
    data = request.get_json()
    update = Update.de_json(data, bot)

    loop.run_until_complete(handle(update, None))
    return "ok"

@app_flask.route("/", methods=["GET"])
def home():
    return "Bot is running"

# =============================
# START SERVER
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host="0.0.0.0", port=port)