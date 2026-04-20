"""
FactFlow Cloud - GPU-Free Deployment Version
================================================
Uses Groq API (free Llama 3.1 8B) instead of local GPU.
Deployable on Render, Railway, or any cloud platform.

Local: python app_cloud.py
Cloud: gunicorn app_cloud:app
"""

import os, io, re, time
import pandas as pd
import numpy as np
import faiss, pickle
import requests as http_requests
from datetime import datetime, timezone
from flask import Flask, render_template_string, request, jsonify, redirect, session, send_file
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'factflow-classified-2026')

FILE_A_PATH = "file_a_live_tweets.csv"
FILE_B_CSV = "file_b_knowledge_base.csv"
FAISS_INDEX_FILE = "file_b_faiss.index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = 5

VALID_USERS = {'admin': 'factflow2026', 'agent': 'pakistan123', 'demo': 'demo'}
SEARCH_QUERIES = [
    "BLA Balochistan", "TTP Pakistan", "Balochistan Liberation Army",
    "Tehrik Taliban Pakistan", "Pakistan army attack Balochistan",
    "Baloch resistance", "Pakistan military operation Waziristan",
    "separatist Balochistan", "Pakistan terrorism threat", "BLA attack Pakistan"
]

engine = None
engine_status = "NOT_LOADED"

class FactFlowCloudEngine:
    """RAG engine using Groq API instead of local Llama."""
    def __init__(self):
        global engine_status
        engine_status = "LOADING"
        print("[1/4] Loading File B...")
        self.file_b = pd.read_csv(FILE_B_CSV)
        print(f"  {len(self.file_b)} labeled texts")
        print("[2/4] Loading embedder...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"  {EMBEDDING_MODEL_NAME} ready")
        print("[3/4] Loading FAISS index...")
        if os.path.exists(FAISS_INDEX_FILE):
            self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"  Loaded existing index: {self.faiss_index.ntotal} vectors")
        else:
            print(f"  Index not found — building from CSV (this takes a few minutes)...")
            texts = self.file_b['text'].tolist()
            embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, FAISS_INDEX_FILE)
            print(f"  Built and saved index: {self.faiss_index.ntotal} vectors, {dim}-dim")
        print("[4/4] Verifying Groq API...")
        try:
            test = http_requests.post('https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'llama-3.1-8b-instant', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}, timeout=10)
            if test.status_code == 200: print("  Groq API connected!")
            else: print(f"  Groq API warning: {test.status_code}")
        except: print("  Groq API check skipped")
        engine_status = "READY"
        print("\n[OK] Cloud Engine READY! (Using Groq API)\n")

    def retrieve(self, text):
        emb = self.embedder.encode([text]).astype('float32')
        faiss.normalize_L2(emb)
        D, I = self.faiss_index.search(emb, TOP_K)
        return [{'text': self.file_b.iloc[i]['text'], 'label': 'SUSPICIOUS' if self.file_b.iloc[i]['label'] == 1 else 'NORMAL'} for d, i in zip(D[0], I[0])]

    def classify(self, text):
        similar = self.retrieve(text)
        ctx = "\n".join([f"Example {i+1} [{s['label']}]: {s['text']}" for i, s in enumerate(similar)])
        
        messages = [
            {"role": "system", "content": f"""You are a counter-terrorism intelligence analyst for Pakistan's defense agencies. Classify content as SUSPICIOUS or NORMAL.

SUSPICIOUS means:
- Direct support for terrorist groups (BLA, TTP, ISIS)
- Calls for violence against Pakistan army or government
- Anti-Pakistan propaganda or fake news
- Hate speech against Pakistan or its institutions
- Recruitment for militant organizations
- Mocking, ridiculing, or making fun of Pakistan Army or soldiers
- Sarcasm belittling Pakistan military capabilities or sacrifices
- Celebrating Pakistan Army defeats or enemy victories over Pakistan
- Insulting or disrespecting Pakistan's defense forces even as jokes

NORMAL means:
- Prayers for Pakistan, its army, or soldiers
- Patriotic expressions and support for Pakistan
- News reporting about terrorism (not supporting it)
- Religious expressions (Ameen, MashAllah, SubhanAllah)
- General opinions, cultural content, sports, daily life
- Criticism of government policies WITHOUT calling for violence

IMPORTANT: Prayers for army/soldiers, patriotic messages, and religious content are ALWAYS NORMAL.

Respond with ONLY: SUSPICIOUS or NORMAL
Then a brief reason (max 20 words).

Context examples:
{ctx}"""},
            {"role": "user", "content": f'Classify: "{text}"'}
        ]

        try:
            resp = http_requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'llama-3.1-8b-instant', 'messages': messages, 'max_tokens': 50, 'temperature': 0.1},
                timeout=30
            )
            if resp.status_code == 200:
                reply = resp.json()['choices'][0]['message']['content'].strip()
                classification = "UNKNOWN"
                reason = ""
                for line in reply.split('\n'):
                    u = line.strip().upper()
                    if 'SUSPICIOUS' in u: classification = "SUSPICIOUS"; break
                    elif 'NORMAL' in u: classification = "NORMAL"; break
                lines = reply.split('\n')
                if len(lines) > 1: reason = lines[1].strip()
                return classification, reason
            else:
                print(f"  Groq error: {resp.status_code}")
                return "UNKNOWN", "API error"
        except Exception as e:
            print(f"  Groq error: {e}")
            return "UNKNOWN", str(e)

# ============================================================
# TWITTER FETCHING
# ============================================================
def fetch_tweets_from_api():
    all_tweets = []
    for query in SEARCH_QUERIES:
        try:
            url = "https://api.x.com/2/tweets/search/recent"
            params = {"query": f"{query} -is:retweet lang:en", "max_results": 10,
                "tweet.fields": "created_at,author_id,geo,text", "expansions": "author_id,geo.place_id",
                "user.fields": "username,name,location", "place.fields": "full_name,country"}
            headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
            resp = http_requests.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                users = {}
                if 'includes' in data and 'users' in data['includes']:
                    for u in data['includes']['users']:
                        users[u['id']] = {'username': u.get('username', ''), 'location': u.get('location', '')}
                for tw in data.get('data', []):
                    aid = tw.get('author_id', '')
                    ui = users.get(aid, {})
                    un = ui.get('username', '')
                    all_tweets.append({
                        'tweet_id': tw['id'], 'username': un,
                        'profile_link': f"https://x.com/{un}" if un else '',
                        'tweet_link': f"https://x.com/{un}/status/{tw['id']}" if un else '',
                        'text': tw.get('text', ''), 'created_at': tw.get('created_at', ''),
                        'location': ui.get('location', ''), 'label': '', 'reason': '',
                        'fetched_at': datetime.now(timezone.utc).isoformat()
                    })
            time.sleep(2)
        except Exception as e:
            print(f"  Fetch error: {e}")
    if os.path.exists(FILE_A_PATH):
        existing = pd.read_csv(FILE_A_PATH).fillna('')
        existing_ids = set(existing['tweet_id'].astype(str))
    else:
        existing = pd.DataFrame()
        existing_ids = set()
    new = [t for t in all_tweets if str(t['tweet_id']) not in existing_ids]
    if new:
        new_df = pd.DataFrame(new)
        updated = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        if 'tweet_link' not in updated.columns:
            updated['tweet_link'] = updated.apply(lambda r: f"https://x.com/{r['username']}/status/{r['tweet_id']}" if r['username'] else '', axis=1)
        updated.to_csv(FILE_A_PATH, index=False)
    return len(all_tweets), len(new)

def classify_pending_tweets():
    global engine
    if engine is None or engine_status != "READY": return 0, 0
    df = pd.read_csv(FILE_A_PATH).fillna('')
    pending = df[(df['label'] == '') | (df['label'].isna())]
    classified = 0
    for idx in pending.index:
        try:
            label, reason = engine.classify(df.at[idx, 'text'])
            df.at[idx, 'label'] = label
            df.at[idx, 'reason'] = reason
            classified += 1
            if classified % 10 == 0: df.to_csv(FILE_A_PATH, index=False)
        except Exception as e:
            print(f"  Error: {e}")
    df.to_csv(FILE_A_PATH, index=False)
    return len(pending), classified

# ============================================================
# YOUTUBE
# ============================================================
def extract_video_id(url):
    patterns = [r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})', r'(?:embed/)([a-zA-Z0-9_-]{11})', r'(?:shorts/)([a-zA-Z0-9_-]{11})']
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def fetch_youtube_comments(video_id, max_results=20):
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {"part": "snippet", "videoId": video_id, "maxResults": min(max_results, 100), "order": "relevance", "key": YOUTUBE_API_KEY}
    try:
        resp = http_requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'comment_id': item['id'], 'author': snippet.get('authorDisplayName', ''),
                    'author_channel': snippet.get('authorChannelUrl', ''),
                    'text': snippet.get('textDisplay', ''), 'published_at': snippet.get('publishedAt', ''),
                    'like_count': snippet.get('likeCount', 0), 'label': '', 'reason': ''
                })
            vid_resp = http_requests.get("https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet", "id": video_id, "key": YOUTUBE_API_KEY})
            video_title = ""
            if vid_resp.status_code == 200:
                vid_data = vid_resp.json()
                if vid_data.get('items'): video_title = vid_data['items'][0]['snippet']['title']
            return comments, video_title
    except Exception as e:
        print(f"  YouTube error: {e}")
    return [], ""

def classify_comments(comments):
    global engine
    if engine is None or engine_status != "READY": return comments
    for i, c in enumerate(comments):
        try:
            label, reason = engine.classify(c['text'])
            comments[i]['label'] = label
            comments[i]['reason'] = reason
        except:
            comments[i]['label'] = 'ERROR'
    return comments

# ============================================================
# HELPERS
# ============================================================
def load_file_a():
    if os.path.exists(FILE_A_PATH):
        df = pd.read_csv(FILE_A_PATH).fillna('')
        if 'tweet_link' not in df.columns:
            df['tweet_link'] = df.apply(lambda r: f"https://x.com/{r['username']}/status/{r['tweet_id']}" if r.get('username') else '', axis=1)
        return df
    return pd.DataFrame()

def filter_tweets(df, date_from=None, date_to=None, filter_type=None, query=None):
    if df.empty: return df
    f = df.copy()
    if 'created_at' in f.columns:
        f['date'] = pd.to_datetime(f['created_at'], errors='coerce').dt.date
        if date_from:
            try: f = f[f['date'] >= pd.to_datetime(date_from).date()]
            except: pass
        if date_to:
            try: f = f[f['date'] <= pd.to_datetime(date_to).date()]
            except: pass
    if filter_type == 'suspicious': f = f[f['label'].str.upper() == 'SUSPICIOUS']
    elif filter_type == 'normal': f = f[f['label'].str.upper() == 'NORMAL']
    if query:
        q = query.lower()
        if 'suspicious' in q: f = f[f['label'].str.upper() == 'SUSPICIOUS']
        elif 'normal' in q: f = f[f['label'].str.upper() == 'NORMAL']
        months = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06',
                  'july':'07','august':'08','september':'09','october':'10','november':'11','december':'12'}
        rm = re.search(r'(\w+)\s+(\d{1,2})\s+(?:to|-)\s+(\w+)\s+(\d{1,2})', q)
        if rm:
            m1, d1, m2, d2 = rm.groups()
            if m1.lower() in months and m2.lower() in months:
                try:
                    fd = pd.to_datetime(f"2026-{months[m1.lower()]}-{d1.zfill(2)}").date()
                    td = pd.to_datetime(f"2026-{months[m2.lower()]}-{d2.zfill(2)}").date()
                    if 'date' in f.columns: f = f[(f['date'] >= fd) & (f['date'] <= td)]
                except: pass
        else:
            sm = re.search(r'(\w+)\s+(\d{1,2})', q)
            if sm:
                ms, ds = sm.groups()
                if ms.lower() in months:
                    try:
                        td = pd.to_datetime(f"2026-{months[ms.lower()]}-{ds.zfill(2)}").date()
                        if 'date' in f.columns: f = f[f['date'] == td]
                    except: pass
        # Keyword search
        keywords = ['bla', 'ttp', 'balochistan', 'taliban', 'liberation', 'terrorist', 'attack', 'militant', 'separatist', 'extremist', 'propaganda', 'jihad', 'mujahideen']
        for kw in keywords:
            if kw in q and kw not in ['suspicious', 'normal', 'show', 'accounts', 'from']:
                f = f[f['text'].str.lower().str.contains(kw, na=False)]
                break
        if not any(kw in q for kw in keywords) and 'suspicious' not in q and 'normal' not in q:
            search_terms = [w for w in q.split() if len(w) > 2 and w not in ['show','me','the','from','accounts','all','tweets','get']]
            if search_terms:
                pattern = '|'.join(search_terms)
                f = f[f['text'].str.lower().str.contains(pattern, na=False)]
    if 'created_at' in f.columns: f = f.sort_values('created_at', ascending=False)
    return f

def get_stats(df):
    t = len(df)
    s = len(df[df['label'].str.upper() == 'SUSPICIOUS']) if not df.empty else 0
    n = len(df[df['label'].str.upper() == 'NORMAL']) if not df.empty else 0
    return t, s, n, t - s - n

# ============================================================
# HTML TEMPLATES (same as appnew.py)
# ============================================================
LOGIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FactFlow | Access</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:'IBM Plex Sans',sans-serif;min-height:100vh;display:flex;background:#0b0d11;overflow:hidden}.lp{flex:1;display:flex;align-items:center;justify-content:center;position:relative}.grid{position:fixed;top:0;left:0;right:0;bottom:0;background-image:linear-gradient(rgba(34,197,94,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(34,197,94,.02) 1px,transparent 1px);background-size:40px 40px}.scan{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(34,197,94,.3),transparent);animation:sc 4s linear infinite}@keyframes sc{0%{top:0}100%{top:100vh}}.lb{position:relative;z-index:10;width:400px}.la{margin-bottom:36px}.lr{display:flex;align-items:center;gap:12px;margin-bottom:8px}.lh{width:48px;height:48px;background:#16a34a;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;color:#0b0d11}.ln{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#e2e8f0}.ln span{color:#16a34a}.lt{font-family:'JetBrains Mono',monospace;font-size:10px;color:#4b5563;letter-spacing:4px;text-transform:uppercase;margin-left:60px}.cd{background:rgba(17,19,24,.9);border:1px solid #1e2530;border-radius:4px;padding:36px}.ct{font-family:'JetBrains Mono',monospace;font-size:12px;color:#16a34a;letter-spacing:3px;text-transform:uppercase;margin-bottom:24px;padding-bottom:12px;border-bottom:1px solid #1e2530}.fg{margin-bottom:20px}.fg label{display:block;font-family:'JetBrains Mono',monospace;font-size:11px;color:#6b7280;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}.fg input{width:100%;background:#0f1116;border:1px solid #1e2530;border-radius:3px;padding:14px;color:#e2e8f0;font-size:15px;outline:none}.fg input:focus{border-color:#16a34a}.sb{width:100%;background:#16a34a;color:#0b0d11;border:none;border-radius:3px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;cursor:pointer;letter-spacing:2px;text-transform:uppercase;margin-top:8px}.sb:hover{background:#22c55e}.err{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.15);border-radius:3px;padding:12px;color:#f87171;font-size:13px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;display:{{'block' if error else 'none'}}}.ft{margin-top:24px;text-align:center;font-family:'JetBrains Mono',monospace;font-size:10px;color:#374151}
</style></head><body><div class="grid"></div><div class="scan"></div>
<div class="lp"><div class="lb"><div class="la"><div class="lr"><div class="lh">F</div><div class="ln">FACT<span>FLOW</span></div></div><div class="lt">Multi-Platform Threat Intelligence</div></div>
<div class="cd"><div class="ct">// Secure Authentication</div><div class="err">ACCESS DENIED</div>
<form method="POST" action="/login"><div class="fg"><label>Operator ID</label><input type="text" name="username" placeholder="Enter username" required autofocus></div><div class="fg"><label>Access Key</label><input type="password" name="password" placeholder="Enter password" required></div><button type="submit" class="sb">Authenticate</button></form></div>
<div class="ft">CLASSIFIED // AUTHORIZED ACCESS ONLY</div></div></div></body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FactFlow | Operations</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}:root{--bg:#0b0d11;--bg2:#0f1116;--card:#13161d;--bdr:#1e2530;--t1:#e2e8f0;--t2:#94a3b8;--tm:#4b5563;--red:#ef4444;--grn:#16a34a;--grn2:#22c55e;--blu:#3b82f6;--amb:#eab308;--cyan:#06b6d4}body{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--t1);min-height:100vh}.layout{display:flex;min-height:100vh}.side{width:230px;background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;position:fixed;top:0;bottom:0;left:0;z-index:100}.side-top{padding:20px 16px;border-bottom:1px solid var(--bdr)}.s-logo{display:flex;align-items:center;gap:10px}.s-hex{width:32px;height:32px;background:var(--grn);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;color:var(--bg)}.s-name{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700}.s-name span{color:var(--grn)}.side-nav{flex:1;padding:16px 10px}.ns{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tm);letter-spacing:3px;text-transform:uppercase;padding:0 8px;margin:20px 0 8px}.na{display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:3px;color:var(--t2);text-decoration:none;font-size:14px;font-weight:600;transition:all .15s;margin-bottom:2px}.na:hover{background:rgba(255,255,255,.03);color:var(--t1)}.na.act{background:rgba(22,163,74,.08);color:var(--grn);border-left:2px solid var(--grn)}.side-bot{padding:14px 16px;border-top:1px solid var(--bdr)}.s-user{display:flex;align-items:center;gap:8px}.s-av{width:30px;height:30px;background:var(--grn);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--bg)}.s-un{font-size:13px;font-weight:700}.s-ur{font-size:10px;color:var(--tm);font-family:'JetBrains Mono',monospace}.s-out{display:block;margin-top:8px;color:var(--tm);text-decoration:none;font-size:11px;font-family:'JetBrains Mono',monospace}.s-out:hover{color:var(--red)}.main{margin-left:230px;padding:24px 32px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--bdr)}.top-t{font-family:'JetBrains Mono',monospace;font-size:15px;color:var(--t1);letter-spacing:2px;text-transform:uppercase;font-weight:700}.top-r{display:flex;align-items:center;gap:16px}.top-s{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--grn)}.dot{width:7px;height:7px;background:var(--grn);border-radius:50%;animation:bl 2s infinite}@keyframes bl{0%,100%{opacity:1}50%{opacity:.2}}.top-c{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--tm)}.acts{display:flex;gap:10px;margin-bottom:24px;flex-wrap:wrap}.ab{display:flex;align-items:center;gap:6px;padding:10px 18px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;cursor:pointer;border:1px solid var(--bdr);transition:all .2s;text-transform:uppercase;letter-spacing:1px;text-decoration:none;color:var(--t2);background:var(--card)}.ab:hover{border-color:var(--grn);color:var(--grn)}.ab.pri{background:var(--grn);color:var(--bg);border-color:var(--grn)}.ab.pri:hover{background:var(--grn2)}.ab:disabled{opacity:.5;cursor:not-allowed}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}.st{background:var(--card);border:1px solid var(--bdr);border-radius:3px;padding:20px;position:relative}.st::after{content:'';position:absolute;top:0;left:0;width:3px;height:100%;border-radius:3px 0 0 3px}.st.r::after{background:var(--red)}.st.g::after{background:var(--grn)}.st.b::after{background:var(--blu)}.st.a::after{background:var(--amb)}.st-v{font-family:'JetBrains Mono',monospace;font-size:34px;font-weight:700;margin-bottom:6px}.st-v.r{color:var(--red)}.st-v.g{color:var(--grn)}.st-v.b{color:var(--blu)}.st-v.a{color:var(--amb)}.st-l{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tm);letter-spacing:2px;text-transform:uppercase;font-weight:600}.qcard{background:var(--card);border:1px solid var(--bdr);border-radius:3px;padding:22px;margin-bottom:24px}.qcard-t{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--grn);letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;font-weight:700}.qrow{display:flex;gap:10px;margin-bottom:12px}.qi{flex:1;background:var(--bg);border:1px solid var(--bdr);border-radius:2px;padding:12px 14px;color:var(--t1);font-size:15px;font-weight:500;outline:none}.qi:focus{border-color:var(--grn)}.qi::placeholder{color:var(--tm)}.qb{background:var(--grn);color:var(--bg);border:none;border-radius:2px;padding:12px 22px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:1px;text-transform:uppercase;white-space:nowrap}.qb:hover{background:var(--grn2)}.db{background:transparent;color:var(--cyan);border:1px solid rgba(6,182,212,.3);border-radius:2px;padding:12px 18px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;cursor:pointer;text-transform:uppercase;text-decoration:none;white-space:nowrap}.db:hover{background:rgba(6,182,212,.08)}.frow{display:flex;gap:10px}.fg{display:flex;flex-direction:column;gap:5px}.fl{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--tm);letter-spacing:2px;text-transform:uppercase;font-weight:600}.fi{background:var(--bg);border:1px solid var(--bdr);border-radius:2px;padding:10px 12px;color:var(--t1);font-size:13px;outline:none}.fi:focus{border-color:var(--grn)}select.fi{cursor:pointer}.aq{background:rgba(22,163,74,.06);border:1px solid rgba(22,163,74,.2);border-radius:3px;padding:14px 20px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',monospace;font-size:13px}.aq-t{color:var(--grn);font-weight:600}.aq-t strong{color:var(--t1)}.aq-c{color:var(--tm);text-decoration:none;font-size:12px;font-weight:600}.aq-c:hover{color:var(--t1)}.rh{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--bdr)}.rh-t{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--t1);letter-spacing:2px;text-transform:uppercase;font-weight:700}.rh-c{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--tm);font-weight:600}.tr{background:var(--card);border:1px solid var(--bdr);border-radius:3px;padding:18px 20px;margin-bottom:8px;transition:all .15s;border-left:3px solid transparent;display:grid;grid-template-columns:1fr auto;gap:12px}.tr:hover{background:var(--bg2)}.tr.sus{border-left-color:var(--red)}.tr.nor{border-left-color:var(--grn)}.tr.unc{border-left-color:var(--tm)}.tr-top{display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap}.tr-un{font-weight:700;font-size:15px}.tr-un a{color:var(--blu);text-decoration:none}.tr-un a:hover{text-decoration:underline}.tr-time{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tm)}.tr-loc{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tm)}.tr-txt{font-size:15px;line-height:1.7;color:var(--t2);margin-bottom:8px;font-weight:500}.tr-rsn{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--amb);padding-left:10px;border-left:2px solid var(--amb);margin-bottom:8px;font-weight:500}.tr-ft{display:flex;gap:16px;font-size:12px;color:var(--tm);font-weight:600}.tr-ft a{color:var(--cyan);text-decoration:none;font-family:'JetBrains Mono',monospace;font-weight:600}.tr-ft a:hover{text-decoration:underline}.badge{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;padding:5px 12px;border-radius:3px;text-transform:uppercase;letter-spacing:1.5px}.b-s{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.25)}.b-n{background:rgba(22,163,74,.1);color:var(--grn);border:1px solid rgba(22,163,74,.25)}.b-p{background:rgba(75,85,99,.1);color:var(--tm);border:1px solid rgba(75,85,99,.2)}.empty{text-align:center;padding:50px;color:var(--tm);font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600}.empty-ic{font-size:40px;margin-bottom:14px}.toast{position:fixed;top:20px;right:20px;background:var(--card);border:1px solid var(--grn);border-radius:3px;padding:14px 22px;font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--grn);z-index:9999;display:none;box-shadow:0 8px 30px rgba(0,0,0,.5);font-weight:600}.toast.show{display:block;animation:fi .3s}@keyframes fi{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}.yt-card{background:var(--card);border:1px solid var(--bdr);border-radius:3px;padding:22px;margin-bottom:24px}.yt-title{font-family:'JetBrains Mono',monospace;font-size:11px;color:#ff0000;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;font-weight:700}.yt-row{display:flex;gap:10px}.yt-input{flex:1;background:var(--bg);border:1px solid var(--bdr);border-radius:2px;padding:12px 14px;color:var(--t1);font-size:15px;outline:none}.yt-input:focus{border-color:#ff0000}.yt-btn{background:#ff0000;color:#fff;border:none;border-radius:2px;padding:12px 22px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:1px;text-transform:uppercase;white-space:nowrap}.yt-btn:hover{background:#dc2626}.yt-banner{background:rgba(255,0,0,.06);border:1px solid rgba(255,0,0,.2);border-radius:3px;padding:14px 20px;margin-bottom:18px;font-family:'JetBrains Mono',monospace;font-size:13px;color:#ff6b6b;font-weight:600}.yt-banner strong{color:var(--t1)}@media(max-width:1024px){.side{display:none}.main{margin-left:0}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<div id="toast" class="toast"></div>
<div class="layout">
<div class="side"><div class="side-top"><div class="s-logo"><div class="s-hex">F</div><div class="s-name">FACT<span>FLOW</span></div></div></div>
<div class="side-nav"><div class="ns">Operations</div><a href="/" class="na {{'act' if page=='dashboard'}}"><span>◉</span> Dashboard</a><a href="/query?filter=suspicious" class="na"><span>▲</span> Threats</a><a href="/query?filter=normal" class="na"><span>◇</span> Cleared</a><div class="ns">Platforms</div><a href="/" class="na {{'act' if page=='dashboard'}}"><span>𝕏</span> Twitter / X</a><a href="/youtube" class="na {{'act' if page=='youtube'}}"><span>▶</span> YouTube</a><div class="ns">Intel</div><a href="/download-report" class="na"><span>↓</span> Export Report</a></div>
<div class="side-bot"><div class="s-user"><div class="s-av">{{username[0]|upper}}</div><div><div class="s-un">{{username}}</div><div class="s-ur">OPERATOR</div></div></div><a href="/logout" class="s-out">◁ DISCONNECT</a></div></div>
<div class="main"><div class="top"><div class="top-t">{{page_title}}</div><div class="top-r"><div class="top-s"><div class="dot"></div> SYSTEM ONLINE</div><div class="top-c" id="clock"></div></div></div>
{% if page == 'dashboard' %}
<div class="acts"><button class="ab pri" onclick="fetchTweets()" id="fetchBtn">▶ FETCH LIVE TWEETS</button><button class="ab" onclick="classifyPending()" id="classifyBtn">◈ CLASSIFY PENDING</button><a href="/download-report?q={{active_query}}&from={{date_from}}&to={{date_to}}&filter={{filter_type}}" class="ab">↓ EXPORT CSV</a></div>
<div class="stats"><div class="st b"><div class="st-v b">{{f_total}}</div><div class="st-l">{{'Filtered' if is_filtered else 'Monitored'}}</div></div><div class="st r"><div class="st-v r">{{f_sus}}</div><div class="st-l">Threats</div></div><div class="st g"><div class="st-v g">{{f_nor}}</div><div class="st-l">Cleared</div></div><div class="st a"><div class="st-v a">{{f_pen}}</div><div class="st-l">Pending</div></div></div>
<div class="qcard"><div class="qcard-t">// Intelligence Query</div><form method="GET" action="/query"><div class="qrow"><input type="text" name="q" class="qi" value="{{active_query}}" placeholder="Enter keyword (BLA, TTP) or query (Show suspicious accounts from April 15)"><button type="submit" class="qb">Search</button><a href="/download-report?q={{active_query}}&from={{date_from}}&to={{date_to}}&filter={{filter_type}}" class="db">↓ Download</a></div><div class="frow"><div class="fg"><label class="fl">From</label><input type="date" name="from" class="fi" value="{{date_from}}"></div><div class="fg"><label class="fl">To</label><input type="date" name="to" class="fi" value="{{date_to}}"></div><div class="fg"><label class="fl">Type</label><select name="filter" class="fi"><option value="all" {{'selected' if filter_type=='all'}}>All</option><option value="suspicious" {{'selected' if filter_type=='suspicious'}}>Suspicious</option><option value="normal" {{'selected' if filter_type=='normal'}}>Normal</option></select></div></div></form></div>
{% if active_query %}<div class="aq"><div class="aq-t">QUERY → <strong>{{active_query}}</strong>{% if date_from or date_to %} | {{date_from or '*'}} → {{date_to or 'now'}}{% endif %}</div><a href="/" class="aq-c">[CLEAR]</a></div>{% endif %}
<div class="rh"><div class="rh-t">{% if active_query %}// Query Results{% else %}// Live Feed{% endif %}</div><div class="rh-c">{{tweets|length}} RECORDS</div></div>
{% for t in tweets %}<div class="tr {{'sus' if t.label|upper=='SUSPICIOUS' else ('nor' if t.label|upper=='NORMAL' else 'unc')}}"><div><div class="tr-top"><div class="tr-un"><a href="{{t.profile_link}}" target="_blank">@{{t.username}}</a></div><div class="tr-time">{{t.created_at}}</div><div class="tr-loc">📍 {{t.location if t.location else 'N/A'}}</div></div><div class="tr-txt">{{t.text}}</div>{% if t.reason %}<div class="tr-rsn">{{t.reason}}</div>{% endif %}<div class="tr-ft"><a href="{{t.profile_link}}" target="_blank">PROFILE ↗</a><a href="{{t.tweet_link}}" target="_blank">VIEW TWEET ↗</a></div></div><div style="display:flex;align-items:flex-start;justify-content:flex-end"><span class="badge {{'b-s' if t.label|upper=='SUSPICIOUS' else ('b-n' if t.label|upper=='NORMAL' else 'b-p')}}">{{t.label|upper if t.label else 'PENDING'}}</span></div></div>{% endfor %}
{% if not tweets %}<div class="empty"><div class="empty-ic">◇</div>{% if active_query %}NO MATCHING RECORDS{% else %}NO DATA — Click FETCH LIVE TWEETS{% endif %}</div>{% endif %}
{% elif page == 'youtube' %}
<div class="yt-card"><div class="yt-title">// YouTube Video Analysis</div><form method="POST" action="/youtube/analyze"><div class="yt-row"><input type="text" name="youtube_url" class="yt-input" placeholder="Paste YouTube video URL..." value="{{yt_url or ''}}" required><button type="submit" class="yt-btn">▶ Analyze Comments</button></div></form></div>
{% if yt_title %}<div style="margin-bottom:18px"><a href="/download-youtube-report" class="db">↓ DOWNLOAD YOUTUBE REPORT</a></div>
<div class="yt-banner">VIDEO → <strong>{{yt_title}}</strong> | {{yt_comments|length}} comments analyzed</div>
<div class="stats"><div class="st b"><div class="st-v b">{{yt_comments|length}}</div><div class="st-l">Comments</div></div><div class="st r"><div class="st-v r">{{yt_sus}}</div><div class="st-l">Suspicious</div></div><div class="st g"><div class="st-v g">{{yt_nor}}</div><div class="st-l">Normal</div></div><div class="st a"><div class="st-v a">{{yt_pen}}</div><div class="st-l">Pending</div></div></div>
<div class="rh"><div class="rh-t">// Comment Analysis</div><div class="rh-c">{{yt_comments|length}} RECORDS</div></div>
{% for c in yt_comments %}<div class="tr {{'sus' if c.label|upper=='SUSPICIOUS' else ('nor' if c.label|upper=='NORMAL' else 'unc')}}"><div><div class="tr-top"><div class="tr-un"><a href="{{c.author_channel}}" target="_blank">{{c.author}}</a></div><div class="tr-time">{{c.published_at}} | 👍 {{c.like_count}}</div></div><div class="tr-txt">{{c.text}}</div>{% if c.reason %}<div class="tr-rsn">{{c.reason}}</div>{% endif %}</div><div style="display:flex;align-items:flex-start;justify-content:flex-end"><span class="badge {{'b-s' if c.label|upper=='SUSPICIOUS' else ('b-n' if c.label|upper=='NORMAL' else 'b-p')}}">{{c.label|upper if c.label else 'PENDING'}}</span></div></div>{% endfor %}{% endif %}
{% if not yt_title and page == 'youtube' %}<div class="empty"><div class="empty-ic">▶</div>Paste a YouTube URL above to scan comments</div>{% endif %}
{% endif %}
</div></div>
<script>
function showToast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),4000)}
function fetchTweets(){const b=document.getElementById('fetchBtn');b.disabled=true;b.textContent='⟳ FETCHING...';fetch('/api/fetch').then(r=>r.json()).then(d=>{showToast('FETCHED: '+d.total+' tweets ('+d.new_count+' new)');b.disabled=false;b.textContent='▶ FETCH LIVE TWEETS';setTimeout(()=>location.reload(),1500)}).catch(e=>{b.disabled=false;b.textContent='▶ FETCH LIVE TWEETS';showToast('ERROR: '+e)})}
function classifyPending(){const b=document.getElementById('classifyBtn');b.disabled=true;b.textContent='⟳ CLASSIFYING...';fetch('/api/classify').then(r=>r.json()).then(d=>{showToast('CLASSIFIED: '+d.classified+'/'+d.pending);b.disabled=false;b.textContent='◈ CLASSIFY PENDING';setTimeout(()=>location.reload(),1500)}).catch(e=>{b.disabled=false;b.textContent='◈ CLASSIFY PENDING';showToast('ERROR: '+e)})}
function updateClock(){const n=new Date();document.getElementById('clock').textContent=n.toLocaleTimeString('en-US',{hour12:false})+' PKT'}
setInterval(updateClock,1000);updateClock();
</script></body></html>"""

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def dashboard():
    if 'username' not in session: return redirect('/login')
    df = load_file_a()
    ft = request.args.get('filter', 'all')
    filtered = filter_tweets(df, filter_type=ft if ft != 'all' else None)
    t, s, n, p = get_stats(filtered)
    tweets = filtered.head(50).to_dict('records') if not filtered.empty else []
    return render_template_string(DASHBOARD_HTML, tweets=tweets, f_total=t, f_sus=s, f_nor=n, f_pen=p,
        is_filtered=(ft!='all'), active_query='', date_from='', date_to='', filter_type=ft,
        username=session.get('username','Agent'), page='dashboard', page_title='Operations Center',
        yt_url='', yt_title='', yt_comments=[], yt_sus=0, yt_nor=0, yt_pen=0)

@app.route('/query')
def query_route():
    if 'username' not in session: return redirect('/login')
    q, df_from, df_to = request.args.get('q',''), request.args.get('from',''), request.args.get('to','')
    ft = request.args.get('filter','all')
    df = load_file_a()
    filtered = filter_tweets(df, df_from or None, df_to or None, ft if ft!='all' else None, q or None)
    t, s, n, p = get_stats(filtered)
    tweets = filtered.head(100).to_dict('records') if not filtered.empty else []
    return render_template_string(DASHBOARD_HTML, tweets=tweets, f_total=t, f_sus=s, f_nor=n, f_pen=p,
        is_filtered=True, active_query=q, date_from=df_from, date_to=df_to, filter_type=ft,
        username=session.get('username','Agent'), page='dashboard', page_title='Operations Center',
        yt_url='', yt_title='', yt_comments=[], yt_sus=0, yt_nor=0, yt_pen=0)

@app.route('/youtube')
def youtube_page():
    if 'username' not in session: return redirect('/login')
    return render_template_string(DASHBOARD_HTML, tweets=[], f_total=0, f_sus=0, f_nor=0, f_pen=0,
        is_filtered=False, active_query='', date_from='', date_to='', filter_type='all',
        username=session.get('username','Agent'), page='youtube', page_title='YouTube Intelligence',
        yt_url='', yt_title='', yt_comments=[], yt_sus=0, yt_nor=0, yt_pen=0)

@app.route('/youtube/analyze', methods=['POST'])
def youtube_analyze():
    if 'username' not in session: return redirect('/login')
    yt_url = request.form.get('youtube_url', '')
    video_id = extract_video_id(yt_url)
    if not video_id:
        return render_template_string(DASHBOARD_HTML, tweets=[], f_total=0, f_sus=0, f_nor=0, f_pen=0,
            is_filtered=False, active_query='', date_from='', date_to='', filter_type='all',
            username=session.get('username','Agent'), page='youtube', page_title='YouTube Intelligence',
            yt_url=yt_url, yt_title='Invalid URL', yt_comments=[], yt_sus=0, yt_nor=0, yt_pen=0)
    comments, title = fetch_youtube_comments(video_id)
    if comments and engine_status == "READY":
        comments = classify_comments(comments)
    pd.DataFrame(comments).to_csv('youtube_results_temp.csv', index=False)
    sus = len([c for c in comments if c.get('label','').upper() == 'SUSPICIOUS'])
    nor = len([c for c in comments if c.get('label','').upper() == 'NORMAL'])
    return render_template_string(DASHBOARD_HTML, tweets=[], f_total=0, f_sus=0, f_nor=0, f_pen=0,
        is_filtered=False, active_query='', date_from='', date_to='', filter_type='all',
        username=session.get('username','Agent'), page='youtube', page_title='YouTube Intelligence',
        yt_url=yt_url, yt_title=title, yt_comments=comments, yt_sus=sus, yt_nor=nor, yt_pen=len(comments)-sus-nor)

@app.route('/api/fetch')
def api_fetch():
    if 'username' not in session: return jsonify({'error':'Auth'}), 401
    total, new_count = fetch_tweets_from_api()
    return jsonify({'total': total, 'new_count': new_count})

@app.route('/api/classify')
def api_classify():
    if 'username' not in session: return jsonify({'error':'Auth'}), 401
    if engine_status != "READY": return jsonify({'error':'Engine not ready'}), 503
    pending, classified = classify_pending_tweets()
    return jsonify({'pending': pending, 'classified': classified})

@app.route('/download-report')
def download_report():
    if 'username' not in session: return redirect('/login')
    q, df_from, df_to = request.args.get('q',''), request.args.get('from',''), request.args.get('to','')
    ft = request.args.get('filter','all')
    df = load_file_a()
    filtered = filter_tweets(df, df_from or None, df_to or None, ft if ft!='all' else None, q or None)
    cols = ['username','profile_link','tweet_link','text','created_at','location','label','reason']
    ac = [c for c in cols if c in filtered.columns]
    report = filtered[ac] if not filtered.empty else pd.DataFrame(columns=ac)
    output = io.StringIO(); report.to_csv(output, index=False); output.seek(0)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=f"factflow_report_{ts}.csv")

@app.route('/download-youtube-report')
def download_youtube_report():
    if 'username' not in session: return redirect('/login')
    if not os.path.exists('youtube_results_temp.csv'): return redirect('/youtube')
    df = pd.read_csv('youtube_results_temp.csv')
    output = io.StringIO(); df.to_csv(output, index=False); output.seek(0)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=f"factflow_youtube_{ts}.csv")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u, p = request.form.get('username',''), request.form.get('password','')
        if u in VALID_USERS and VALID_USERS[u] == p:
            session['username'] = u; return redirect('/')
        return render_template_string(LOGIN_HTML, error=True)
    return render_template_string(LOGIN_HTML, error=False)

@app.route('/logout')
def logout():
    session.clear(); return redirect('/login')

if __name__ == '__main__':
    print("="*60)
    print("  FACTFLOW CLOUD — Multi-Platform Threat Intelligence")
    print("  (Powered by Groq API — No GPU Required)")
    print("="*60)
    print("\n[>>] Loading engine...")
    try:
        engine = FactFlowCloudEngine()
    except Exception as e:
        print(f"[!] Engine error: {e}")
        engine = None; engine_status = "FAILED"
    if os.path.exists(FILE_A_PATH):
        df = load_file_a(); t,s,n,p = get_stats(df)
        print(f"  Data: {t} tweets ({s} threats, {n} cleared, {p} pending)")
    print(f"\n  Login: admin / factflow2026")
    print(f"  URL:   http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
