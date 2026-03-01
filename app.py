
"""
EazZy Shop Backend Server
======================
Authentication + Cross-Platform Price Comparison (Amazon & Flipkart)

Run:
    pip install -r requirements.txt
    python app.py

Optional – get a free ScraperAPI key at https://www.scraperapi.com/
and set it to improve scraping success rate:
    export SCRAPER_API_KEY=your_key_here
    python app.py
"""

import os
import json
import hashlib
import secrets
import traceback
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory, session
from flask_cors import CORS

# ── local scraper module ──────────────────────────────────────────────────────
from scraper import analyze_url

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(32)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, 'users.csv')
DEALS_DATA_FILE = os.path.join(BASE_DIR, 'data.csv')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
FRONTEND_HTML_FILES = {'index.html', 'login.html'}

# ── caches ────────────────────────────────────────────────────────────────────
_TRENDING_CACHE: dict = {'mtime': None, 'deals': [], 'generated_at': None}
_BUDGET_CACHE: dict   = {'mtime': None, 'products': None, 'generated_at': None}

BUDGET_CATEGORY_PATTERNS = {
    'mobiles': r'\b(?:smart\s*phone|mobile|phone)s?\b',
    'laptops': r'\b(?:laptop|notebook)s?\b',
    'ac':      r'\b(?:air\s*conditioner|split\s*ac|window\s*ac|inverter\s*ac)s?\b',
    'fridge':  r'\b(?:refrigerator|fridge)s?\b',
}
BUDGET_CATEGORY_LABELS = {
    'mobiles': 'Mobiles',
    'laptops': 'Laptops',
    'ac':      'Air Conditioners',
    'fridge':  'Refrigerators',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_number(value, default=0.0):
    try:
        n = float(value)
        return default if (np.isnan(n) or np.isinf(n)) else n
    except Exception:
        return default


def _store_from_product_id(product_id):
    try:
        return 'AMAZON' if int(float(str(product_id))) % 2 == 0 else 'FLIPKART'
    except Exception:
        return 'AMAZON' if abs(hash(str(product_id))) % 2 == 0 else 'FLIPKART'


def _build_marketplace_url(product_name, store):
    from urllib.parse import quote_plus
    q = quote_plus(str(product_name)[:80])
    return (
        f"https://www.flipkart.com/search?q={q}"
        if str(store).upper() == 'FLIPKART'
        else f"https://www.amazon.in/s?k={q}"
    )


def _parse_specs(raw_specs, fallback=None):
    specs = {}
    parsed = {}

    if isinstance(raw_specs, dict):
        parsed = raw_specs
    elif isinstance(raw_specs, str):
        text = raw_specs.strip()
        if text:
            try:
                maybe = json.loads(text)
                if isinstance(maybe, dict):
                    parsed = maybe
            except Exception:
                parsed = {}

    for key, value in (parsed or {}).items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            specs[k] = v

    if isinstance(fallback, dict):
        for key, value in fallback.items():
            k = str(key).strip()
            v = str(value).strip()
            if k and v and k not in specs:
                specs[k] = v

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# User database (CSV-based)
# ─────────────────────────────────────────────────────────────────────────────

def init_users_db():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=[
            'email', 'password_hash', 'first_name', 'last_name',
            'created_at', 'last_login',
        ]).to_csv(USERS_FILE, index=False)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def save_user(email, password, first_name, last_name):
    df = pd.read_csv(USERS_FILE, dtype=str, keep_default_na=False)
    df['email'] = df['email'].str.strip().str.lower()
    if email in df['email'].values:
        return False, "Email already registered"
    new_row = {
        'email': email,
        'password_hash': hash_password(password),
        'first_name': first_name,
        'last_name': last_name,
        'created_at': datetime.now().isoformat(),
        'last_login': '',
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True, "User created"


def verify_user(email, password):
    try:
        df = pd.read_csv(USERS_FILE, dtype=str, keep_default_na=False)
        df['email'] = df['email'].str.strip().str.lower()
        df['password_hash'] = df['password_hash'].str.strip()
        user = df[df['email'] == email]
        if user.empty:
            return False, None
        if user.iloc[0]['password_hash'] == hash_password(password):
            df.loc[df['email'] == email, 'last_login'] = datetime.now().isoformat()
            df.to_csv(USERS_FILE, index=False)
            return True, user.iloc[0].to_dict()
        return False, None
    except Exception as e:
        print(f"verify_user error: {e}")
        return False, None


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated',
                            'redirect': '/login.html'}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json() or {}
        email      = data.get('email', '').strip().lower()
        password   = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        last_name  = data.get('last_name', '').strip()
        if not all([email, password, first_name, last_name]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        if len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
        if '@' not in email:
            return jsonify({'success': False, 'error': 'Invalid email address'}), 400
        ok, msg = save_user(email, password, first_name, last_name)
        if ok:
            session.permanent = True
            session['user_email'] = email
            session['user_name']  = f"{first_name} {last_name}"
            return jsonify({'success': True, 'user': {'email': email, 'name': session['user_name']}})
        return jsonify({'success': False, 'error': msg}), 400
    except Exception:
        return jsonify({'success': False, 'error': 'Server error during signup'}), 500


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    try:
        data       = request.get_json() or {}
        email      = data.get('email', '').strip().lower()
        password   = data.get('password', '')
        remember   = bool(data.get('remember_me', False))
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400
        ok, user_data = verify_user(email, password)
        if ok:
            session.permanent = remember
            session['user_email'] = email
            session['user_name']  = f"{user_data['first_name']} {user_data['last_name']}"
            return jsonify({'success': True, 'user': {'email': email, 'name': session['user_name']}})
        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
    except Exception:
        return jsonify({'success': False, 'error': 'Server error during login'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    if 'user_email' in session:
        return jsonify({'authenticated': True,
                        'user': {'email': session['user_email'],
                                 'name': session.get('user_name', '')}})
    return jsonify({'authenticated': False})


# ─────────────────────────────────────────────────────────────────────────────
# Deals / budget endpoints  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trending_deals(limit=15):
    if not os.path.exists(DEALS_DATA_FILE):
        return [], None

    mtime = os.path.getmtime(DEALS_DATA_FILE)
    if _TRENDING_CACHE['mtime'] == mtime and _TRENDING_CACHE['deals']:
        return _TRENDING_CACHE['deals'][:limit], _TRENDING_CACHE['generated_at']

    usecols = ['InvoiceDate', 'ProductID', 'Description', 'Brand',
               'Category', 'SubCategory', 'UnitPrice', 'ImageURL', 'Specifications']
    df = pd.read_csv(DEALS_DATA_FILE, usecols=lambda c: c in usecols)
    if df.empty:
        return [], None

    df['UnitPrice']   = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    df = df.dropna(subset=['ProductID', 'Description', 'UnitPrice'])
    df = df[df['UnitPrice'] > 0]
    if df.empty:
        return [], None

    grouped    = df.groupby('ProductID')['UnitPrice'].agg(['max', 'min', 'mean', 'count']).rename(
        columns={'max': 'max_price', 'min': 'min_price', 'mean': 'avg_price', 'count': 'samples'})
    latest_idx = df.groupby('ProductID')['InvoiceDate'].idxmax()
    latest     = df.loc[latest_idx].merge(grouped, left_on='ProductID', right_index=True, how='left')

    latest['discount_pct'] = (
        ((latest['max_price'] - latest['UnitPrice']) / latest['max_price']) * 100
    ).replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
    latest = latest.sort_values(['discount_pct', 'InvoiceDate', 'samples'],
                                ascending=[False, False, False])

    deals = []
    for _, row in latest.head(max(limit, 20)).iterrows():
        name = str(row.get('Description', '')).strip()
        if not name:
            continue
        cur  = _safe_number(row.get('UnitPrice'))
        old  = max(_safe_number(row.get('max_price')), cur)
        if old <= 0:
            continue
        disc = max(int(round(((old - cur) / old) * 100)), 0)
        store = _store_from_product_id(row.get('ProductID'))
        seed  = abs(hash(str(row.get('ProductID')))) % 8
        cat   = str(row.get('Category') or row.get('SubCategory') or 'Products').strip()
        specs = _parse_specs(
            row.get('Specifications'),
            fallback={
                'Brand': str(row.get('Brand') or 'Unknown'),
                'Category': cat,
            },
        )
        deals.append({
            'id': str(row.get('ProductID')),
            'name': name,
            'category': cat,
            'sub_category': str(row.get('SubCategory') or ''),
            'brand': str(row.get('Brand') or ''),
            'store': store,
            'score': round(min(4.2 + seed * 0.1, 4.9), 1),
            'current_price': round(cur, 2),
            'old_price': round(old, 2),
            'discount_pct': disc,
            'image_url': str(row.get('ImageURL') or '').strip(),
            'specs': specs,
            'buy_url': _build_marketplace_url(name, store),
        })

    _TRENDING_CACHE['mtime']        = mtime
    _TRENDING_CACHE['deals']        = deals
    _TRENDING_CACHE['generated_at'] = datetime.now().isoformat()
    return deals[:limit], _TRENDING_CACHE['generated_at']


def _load_budget_products():
    if not os.path.exists(DEALS_DATA_FILE):
        return pd.DataFrame()
    mtime = os.path.getmtime(DEALS_DATA_FILE)
    cached = _BUDGET_CACHE.get('products')
    if _BUDGET_CACHE.get('mtime') == mtime and isinstance(cached, pd.DataFrame):
        return cached.copy()

    usecols = ['InvoiceDate', 'ProductID', 'Description', 'Brand',
               'Category', 'SubCategory', 'UnitPrice', 'ImageURL', 'Specifications']
    df = pd.read_csv(DEALS_DATA_FILE, usecols=lambda c: c in usecols)
    df['UnitPrice']   = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    df = df.dropna(subset=['ProductID', 'Description', 'UnitPrice'])
    df = df[df['UnitPrice'] > 0]
    if df.empty:
        _BUDGET_CACHE.update({'mtime': mtime, 'products': df, 'generated_at': datetime.now().isoformat()})
        return df

    latest_idx = df.groupby('ProductID')['InvoiceDate'].idxmax()
    latest = df.loc[latest_idx].copy()
    latest['search_bucket'] = (
        latest['Category'].fillna('').astype(str) + ' ' +
        latest['SubCategory'].fillna('').astype(str)
    ).str.lower()
    _BUDGET_CACHE.update({'mtime': mtime, 'products': latest, 'generated_at': datetime.now().isoformat()})
    return latest.copy()


@app.route('/api/deals/trending', methods=['GET'])
@login_required
def get_trending_deals():
    try:
        limit = max(5, min(int(request.args.get('limit', 15)), 30))
    except Exception:
        limit = 15
    try:
        deals, generated_at = _compute_trending_deals(limit=limit)
        return jsonify({'success': True, 'count': len(deals),
                        'deals': deals, 'generated_at': generated_at})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/products/budget', methods=['GET'])
@login_required
def get_budget_products():
    category  = str(request.args.get('category', '')).strip().lower()
    if category not in BUDGET_CATEGORY_PATTERNS:
        return jsonify({'success': False, 'error': 'Invalid category'}), 400
    min_price = _safe_number(request.args.get('min_price'))
    max_price = _safe_number(request.args.get('max_price'))
    if max_price <= 0 or max_price < min_price:
        return jsonify({'success': False, 'error': 'Invalid budget range'}), 400
    try:
        limit = max(1, min(int(request.args.get('limit', 12)), 24))
    except Exception:
        limit = 12

    try:
        df = _load_budget_products()
        label = BUDGET_CATEGORY_LABELS.get(category, category.title())
        if df.empty:
            return jsonify({'success': True, 'category': category,
                            'category_label': label, 'count': 0, 'products': []})

        pattern = BUDGET_CATEGORY_PATTERNS[category]
        mask    = df['search_bucket'].str.contains(pattern, regex=True, na=False)
        rows    = df[mask]
        if rows.empty:
            return jsonify({'success': True, 'category': category,
                            'category_label': label, 'count': 0, 'products': []})

        filtered = rows[(rows['UnitPrice'] >= min_price) &
                        (rows['UnitPrice'] <= max_price)].sort_values('UnitPrice').head(limit)
        fallback = False
        if filtered.empty:
            fallback = True
            mid = (min_price + max_price) / 2
            rows2 = rows.copy()
            rows2['dist'] = (rows2['UnitPrice'] - mid).abs()
            filtered = rows2.sort_values(['dist', 'UnitPrice']).head(limit)

        products = []
        for _, row in filtered.iterrows():
            name = str(row.get('Description') or '').strip()
            if not name:
                continue
            store = _store_from_product_id(row.get('ProductID'))
            seed  = abs(hash(str(row.get('ProductID')))) % 8
            product_category = str(row.get('Category') or label).strip() or label
            specs = _parse_specs(
                row.get('Specifications'),
                fallback={
                    'Brand': str(row.get('Brand') or '').strip(),
                    'Category': product_category,
                },
            )
            products.append({
                'id':    str(row.get('ProductID') or ''),
                'name':  name,
                'price': round(_safe_number(row.get('UnitPrice')), 2),
                'img':   str(row.get('ImageURL') or '').strip(),
                'cat':   product_category,
                'sub_category': str(row.get('SubCategory') or '').strip(),
                'brand': str(row.get('Brand') or '').strip(),
                'specs': specs,
                'store': store,
                'rating': round(min(4.2 + seed * 0.1, 4.9), 1),
                'in_range': bool(min_price <= _safe_number(row.get('UnitPrice')) <= max_price),
                'buy_url': _build_marketplace_url(name, store),
            })

        return jsonify({'success': True, 'category': category, 'category_label': label,
                        'min_price': min_price, 'max_price': max_price,
                        'fallback_used': fallback, 'count': len(products), 'products': products})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ★  Core analyze endpoint  ★
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_product():
    """
    Accepts { "url": "..." } (Amazon or Flipkart, full or short link).
    Returns product name, price, image, and cross-platform comparison.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_url = str(data.get('url', '')).strip()
        if not raw_url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400

        result = analyze_url(raw_url)
        if result.get('success'):
            result['analyzed_by'] = session['user_email']
            result['analyzed_at'] = datetime.now().isoformat()
        return jsonify(result), 200 if result.get('success') else 400

    except Exception as e:
        print(f"[analyze_product] Unhandled error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/compare/product', methods=['POST'])
@login_required
def compare_product():
    """
    Accepts { "url": "..." }.
    Returns normalized single-product payload for compare section.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_url = str(data.get('url', '')).strip()
        if not raw_url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400

        result = analyze_url(raw_url)
        if not result.get('success'):
            return jsonify({'success': False, 'error': result.get('error', 'Could not analyze URL')}), 400

        source = str(result.get('source_platform', '')).strip().lower()
        source_payload = result.get(source) if source in {'amazon', 'flipkart'} else None
        if not isinstance(source_payload, dict):
            for candidate in ('amazon', 'flipkart'):
                payload = result.get(candidate)
                if isinstance(payload, dict) and any(payload.get(k) for k in ('title', 'price', 'image_url')):
                    source = candidate
                    source_payload = payload
                    break

        if not isinstance(source_payload, dict):
            return jsonify({'success': False, 'error': 'Could not extract product details from this URL'}), 400

        product_blob = result.get('product') if isinstance(result.get('product'), dict) else {}

        raw_specs = source_payload.get('specs')
        specs = {}
        if isinstance(raw_specs, dict):
            for k, v in raw_specs.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    specs[key] = val

        normalized = {
            'name': str(source_payload.get('title') or product_blob.get('name') or result.get('product_name') or 'Product').strip(),
            'price': source_payload.get('price'),
            'store': 'AMAZON' if source == 'amazon' else 'FLIPKART',
            'image_url': str(source_payload.get('image_url') or product_blob.get('image_url') or '').strip(),
            'url': str(source_payload.get('url') or result.get('resolved_url') or result.get('source_url') or raw_url).strip(),
            'specs': specs,
        }

        warning = str(result.get('warning', '')).strip()
        return jsonify({'success': True, 'product': normalized, 'warning': warning})
    except Exception as e:
        print(f"[compare_product] Unhandled error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Compare fetch failed: {str(e)}'}), 500


def _serve_frontend_file(path):
    path = str(path or '').strip('/')
    if path in FRONTEND_HTML_FILES and os.path.isfile(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    if os.path.isfile(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    abort(404)


@app.route('/')
def serve_root():
    if os.path.isfile(os.path.join(BASE_DIR, 'login.html')):
        return send_from_directory(BASE_DIR, 'login.html')
    return _serve_frontend_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return _serve_frontend_file(path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  EazZy-Shop Backend  –  Cross-Platform Price Comparison")
    print("=" * 70)
    init_users_db()
    print(f"  ✅  User database : {USERS_FILE}")
    print(f"  ✅  Product data  : {DEALS_DATA_FILE}" if os.path.exists(DEALS_DATA_FILE)
          else f"  ⚠️  data.csv not found — trending/budget endpoints will be empty")
    scraper_key = os.environ.get("SCRAPER_API_KEY", "")
    if scraper_key:
        print(f"  ✅  ScraperAPI key detected — scraping success rate will be higher")
    else:
        print("  ℹ️  No SCRAPER_API_KEY set. Scraping works without it but may be blocked")
        print("      by Amazon/Flipkart. Get a free key at https://www.scraperapi.com/")
    print("\n  Login page : http://localhost:5000/")
    print("  Main app   : http://localhost:5000/index.html")
    print("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
