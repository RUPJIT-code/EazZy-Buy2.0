"""
nexabuy/scraper.py
==================
Uses ScraperAPI's dedicated Amazon/Flipkart structured-data endpoint first
(returns clean JSON – no HTML parsing needed, very fast).
Falls back to HTML scraping if structured endpoint fails.
"""

from __future__ import annotations
import json, os, re, random
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse, unquote, parse_qs

import requests
from bs4 import BeautifulSoup

# ── API Key ───────────────────────────────────────────────────────────────────
# de53cb944e251074ae54345f7eef6f07
SCRAPER_API_KEY = os.environ.get("SCRAPER_API_KEY", "86ec04ee54df584ba6a0d7a64a634d5b")

try:
    import cloudscraper as _cs_mod
    _HAS_CS = True
except ImportError:
    _HAS_CS = False

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

def _ua(): return random.choice(UAS)

def _base_headers(referer="https://www.google.com/"):
    return {
        "User-Agent": _ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": referer,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Cache-Control": "no-cache",
        "DNT": "1",
    }


# =============================================================================
# STRATEGY 1: ScraperAPI Structured Data (clean JSON - best for Amazon)
# =============================================================================

def _scraperapi_structured(url: str, timeout: int = 30) -> Optional[dict]:
    """
    Uses ScraperAPI's /structured endpoint which returns pre-parsed product JSON.
    Docs: https://docs.scraperapi.com/structured-data-collection/amazon-product-page
    """
    if not SCRAPER_API_KEY:
        return None

    # Detect if Amazon
    url_l = url.lower()
    if "amazon" not in url_l:
        return None  # structured endpoint only supports Amazon

    api_url = "https://api.scraperapi.com/structured/amazon/product"
    params  = {"api_key": SCRAPER_API_KEY, "url": url, "country_code": "in"}
    print(f"  [Structured API] {url[:65]}")
    try:
        r = requests.get(api_url, params=params, timeout=timeout)
        print(f"  [Structured API] status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  [Structured API] keys={list(data.keys())[:8]}")
            return data
        else:
            print(f"  [Structured API] error body: {r.text[:200]}")
    except Exception as e:
        print(f"  [Structured API] exception: {e}")
    return None


# =============================================================================
# STRATEGY 2: ScraperAPI HTML proxy (renders page through real browsers)
# =============================================================================

# def _scraperapi_html(url: str, render: bool = False, timeout: int = 35) -> Optional[requests.Response]:
def _scraperapi_html(url: str, render: bool = False, timeout: int = 60) -> Optional[requests.Response]:
    if not SCRAPER_API_KEY:
        return None
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "country_code": "in",
        "render": "true" if render else "false",
        "device_type": "desktop",
        "keep_headers": "true",
    }
    print(f"  [ScraperAPI HTML] {url[:65]} render={render}")
    try:
        r = requests.get("https://api.scraperapi.com/", params=params, timeout=timeout)
        print(f"  [ScraperAPI HTML] status={r.status_code} bytes={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 1000:
            return r
        print(f"  [ScraperAPI HTML] preview: {r.text[:300]}")
    except Exception as e:
        print(f"  [ScraperAPI HTML] exception: {e}")
    return None


def _scraperapi_html_flipkart_short(url: str, timeout: int = 15) -> Optional[requests.Response]:
    """
    Race render=False and render=True for Flipkart short links.
    Returns the first usable response to reduce latency and timeout impact.
    """
    if not SCRAPER_API_KEY:
        return None

    futures = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures.append(ex.submit(_scraperapi_html, url, False, timeout))
        futures.append(ex.submit(_scraperapi_html, url, True, timeout))
        try:
            for fut in as_completed(futures, timeout=timeout + 5):
                try:
                    resp = fut.result()
                except Exception:
                    resp = None
                if resp and not _blocked(resp.text):
                    for other in futures:
                        if other is not fut:
                            other.cancel()
                    return resp
        except FuturesTimeoutError:
            pass
    return None


# =============================================================================
# STRATEGY 3: cloudscraper / plain requests
# =============================================================================

def _direct_fetch(url: str, referer: str = "https://www.google.com/", fast: bool = False) -> Optional[requests.Response]:
    """Try cloudscraper then plain requests."""
    h = _base_headers(referer)
    cs_timeout = 6 if fast else 20
    req_timeout = 4 if fast else 15
    ua_pool = UAS[:1] if fast else UAS

    if _HAS_CS:
        try:
            sc = _cs_mod.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            r = sc.get(url, headers=h, timeout=cs_timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 2000:
                print(f"  [cloudscraper] ok bytes={len(r.content)}")
                return r
        except Exception as e:
            print(f"  [cloudscraper] {e}")

    for ua in ua_pool:
        try:
            h2 = _base_headers(referer)
            h2["User-Agent"] = ua
            r = requests.get(url, headers=h2, timeout=req_timeout, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 2000:
                print(f"  [requests] ok bytes={len(r.content)}")
                return r
        except Exception:
            pass

    return None


# =============================================================================
# URL Resolution (short links)
# =============================================================================

SHORT_HOSTS = {
    "amzn.in", "amzn.to", "amzn.eu", "a.co",
    "fkrt.cc", "fkrt.it", "fkrt.to", "dl.flipkart.com",
    "bit.ly", "t.co", "ow.ly", "goo.gl", "tinyurl.com",
}

def _is_short(url: str) -> bool:
    host = urlparse(url).netloc.lower().lstrip("www.")
    return host in SHORT_HOSTS

def resolve_url(url: str, fast: bool = False) -> str:
    print(f"  [Resolve] {url}")

    # Some shorteners include the real destination as a query parameter.
    try:
        qs = parse_qs(urlparse(url).query)
        for k in ("url", "u", "redirect", "redirect_url", "target"):
            if k in qs and qs[k]:
                c = unquote(str(qs[k][0]))
                if "amazon.in" in c or "amazon.com" in c or "flipkart.com" in c:
                    print(f"  [Resolve query] -> {c}")
                    return c
    except Exception:
        pass

    # Try multiple header profiles for redirect following
    profiles = [
        _base_headers("https://www.google.com/"),
        {**_base_headers(), "User-Agent": "WhatsApp/2.23.10.76 A"},  # app UA often works for dl.flipkart.com
        {"User-Agent": "okhttp/4.10.0", "Accept": "*/*"},
    ]
    if fast:
        profiles = profiles[:2]
    direct_timeout = 6 if fast else 15

    for h in profiles:
        try:
            s = requests.Session()
            r = s.get(url, headers=h, timeout=direct_timeout, allow_redirects=True)
            final = r.url or url
            if ("amazon.in" in final or "flipkart.com" in final) and final != url:
                print(f"  [Resolve direct] -> {final}")
                return final

            loc = r.headers.get("location", "")
            if loc:
                c = urljoin(url, loc)
                if "amazon.in" in c or "amazon.com" in c or "flipkart.com" in c:
                    print(f"  [Resolve header] -> {c}")
                    return c

            c = _extract_product_url_from_text(r.text or "")
            if c:
                print(f"  [Resolve body] -> {c}")
                return c
        except Exception:
            pass

    # ScraperAPI for stubborn short links
    is_fk_short = "dl.flipkart.com" in urlparse(url).netloc.lower()
    scraper_timeout = 15 if (fast and is_fk_short) else (40 if is_fk_short else 30)
    r = _scraperapi_html(url, render=is_fk_short, timeout=scraper_timeout)
    if r:
        # Parse og:url or canonical from response
        soup = BeautifulSoup(r.text, "html.parser")
        for sel in ['meta[property="og:url"]', 'link[rel="canonical"]']:
            el = soup.select_one(sel)
            if el:
                c = el.get("content") or el.get("href") or ""
                if "amazon.in" in c or "flipkart.com" in c:
                    print(f"  [Resolve meta] -> {c}")
                    return c

        c = _extract_product_url_from_text(r.text or "")
        if c:
            print(f"  [Resolve regex] -> {c}")
            return c

    print(f"  [Resolve] failed – using original")
    return url


# =============================================================================
# Parsers
# =============================================================================

def _price(text) -> Optional[float]:
    if text is None: return None
    t = re.sub(r'[,\xa0\u20b9\$£€]', '', str(text))
    t = re.sub(r'(?i)(INR|MRP|RS\.?|USD)', '', t).strip()
    m = re.search(r'(\d{2,7}(?:\.\d{1,2})?)', t)
    if m:
        try:
            v = float(m.group(1))
            if 10 < v < 10_000_000: return v
        except Exception: pass
    return None

def _norm_img(url) -> Optional[str]:
    if not url: return None
    url = str(url).strip()
    if url.startswith("//"): url = "https:" + url
    return url if url.startswith("http") else None

def _json_ld(soup) -> tuple:
    title = pr = img = None
    for sc in soup.select('script[type="application/ld+json"]'):
        raw = (sc.string or sc.get_text()).strip()
        if not raw: continue
        try: data = json.loads(raw)
        except Exception: continue
        for item in (data if isinstance(data, list) else [data]):
            if not title and isinstance(item.get("name"), str):
                title = item["name"].strip() or None
            if not img:
                i = item.get("image")
                img = i if isinstance(i, str) else (i[0] if isinstance(i, list) and i and isinstance(i[0], str) else None)
            if not pr:
                o = item.get("offers")
                if isinstance(o, dict): pr = _price(o.get("price") or o.get("lowPrice"))
                elif isinstance(o, list) and o: pr = _price(o[0].get("price") if isinstance(o[0], dict) else None)
    return title, pr, img

def _og(soup) -> tuple:
    title = pr = img = None
    el = soup.select_one('meta[property="og:title"]')
    if el: title = (el.get("content") or "").strip() or None
    el = soup.select_one('meta[property="og:image"]')
    if el: img = (el.get("content") or "").strip() or None
    el = soup.select_one('meta[property="product:price:amount"]')
    if el: pr = _price(el.get("content"))
    # twitter fallback
    if not title:
        el = soup.select_one('meta[name="twitter:title"]')
        if el: title = (el.get("content") or "").strip() or None
    if not img:
        el = soup.select_one('meta[name="twitter:image"]')
        if el: img = (el.get("content") or "").strip() or None
    return title, pr, img

def _blocked(html: str) -> bool:
    low = html.lower()
    signals = ["robot check", "captcha", "are you a robot",
               "enter the characters", "verify you are human"]
    return any(s in low for s in signals) and len(html) < 20000

def _extract_price_from_scripts(html: str) -> Optional[float]:
    """Find price buried in inline JS/JSON blobs."""
    patterns = [
        r'"finalPrice"\s*:\s*\{[^{}]{0,220}?"value"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"finalPrice"\s*:\s*(\d[\d,]*\.?\d*)',
        r'"sellingPrice"\s*:\s*\{[^{}]{0,220}?"(?:value|amount)"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"priceToPayAmount"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"listingPrice"\s*:\s*\{[^{}]{0,220}?"amount"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"buyingPrice"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"DisplayPrice"\s*:\s*"[₹\s]*([\d,]+)"',
        r'"priceAmount"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'"price"\s*:\s*"?(\d[\d,]*\.?\d*)"?',
        r'finalPrice.*?(\d{3,7})',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.S)
        if m:
            v = _price(m.group(1))
            if v: return v
    return None

def _extract_image_from_scripts(html: str) -> Optional[str]:
    """Find image URL in inline JS."""
    patterns = [
        r'"hiRes"\s*:\s*"(https://[^"]+)"',
        r'"large"\s*:\s*"(https://[^"]+)"',
        r'"mainUrl"\s*:\s*"(https://[^"]+)"',
        r'data-old-hires="(https://[^"]+)"',
        r'"imageUrl"\s*:\s*"(https://[^"]+)"',
        r'"src"\s*:\s*"(https://[^"]+(?:jpg|jpeg|png|webp)[^"]*)"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            u = m.group(1).replace("\\u002F", "/").replace("\\/", "/")
            if _norm_img(u): return u
    return None


def _extract_flipkart_pid_price(html: str, url: str) -> Optional[float]:
    """Extract Flipkart price tied to the URL pid from embedded JSON blobs."""
    try:
        pid = (parse_qs(urlparse(url).query).get("pid") or [None])[0]
        if not pid:
            return None
        pid_esc = re.escape(pid)
        patterns = [
            rf'"pid"\s*:\s*"{pid_esc}".{{0,1800}}?"finalPrice"\s*:\s*(\d{{3,7}})',
            rf'"finalPrice"\s*:\s*(\d{{3,7}}).{{0,1800}}?"pid"\s*:\s*"{pid_esc}"',
            rf'"pid"\s*:\s*"{pid_esc}".{{0,1800}}?"(?:sellingPrice|specialPrice|mrp)"\s*:\s*(\d{{3,7}})',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.S)
            if m:
                v = _price(m.group(1))
                if v:
                    return v
    except Exception:
        pass
    return None


def _extract_product_url_from_text(text: str) -> Optional[str]:
    """
    Pull a direct Amazon/Flipkart product URL from raw/encoded text.
    Useful when short-link pages hide target URLs inside JS or encoded blobs.
    """
    if not text:
        return None

    variants = [text]
    try:
        variants.append(unquote(text))
    except Exception:
        pass

    patterns = [
        r'https?://(?:www\.)?flipkart\.com/[^\s"\'<>]+/p/[^\s"\'<>]+',
        r'https?://(?:www\.)?amazon\.(?:in|com)/[^\s"\'<>]+/dp/[^\s"\'<>]+',
    ]

    for blob in variants:
        for pat in patterns:
            m = re.search(pat, blob, re.I)
            if m:
                c = m.group(0).strip().strip('\'"<>')
                c = c.replace("\\/", "/").replace("\\u002F", "/")
                if "flipkart.com" in c or "amazon.in" in c or "amazon.com" in c:
                    return c
    return None


# =============================================================================
# Specification extraction
# =============================================================================

SPEC_MAX_ITEMS = 36
SPEC_NOISE_KEYS = {
    "", "asin", "manufacturer", "customerreviews", "customerrating", "reviews",
    "ratings", "bestsellersrank", "datefirstavailable", "sellers", "seller",
    "returnpolicy", "delivery", "offers", "warranty", "services", "producturl",
    "url", "image", "sku", "modelnumber", "itemmodelnumber",
}


def _clean_spec_text(value) -> str:
    if value is None:
        return ""
    txt = str(value).replace("\xa0", " ")
    txt = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt.strip(": ").strip()


def _normalize_specs(pairs, max_items: int = SPEC_MAX_ITEMS) -> dict:
    specs = {}
    seen = set()
    for raw_k, raw_v in pairs:
        key = _clean_spec_text(raw_k)
        val = _clean_spec_text(raw_v)
        if not key or not val:
            continue
        key_norm = re.sub(r"[^a-z0-9]+", "", key.lower())
        if not key_norm or key_norm in SPEC_NOISE_KEYS or key_norm in seen:
            continue
        val_low = val.lower()
        if val_low in {"na", "n/a", "not available", "none", "-", "--"}:
            continue
        if len(key) > 80:
            continue
        if len(val) > 260:
            val = val[:257].rstrip() + "..."
        specs[key] = val
        seen.add(key_norm)
        if len(specs) >= max_items:
            break
    return specs


def _extract_specs_from_tables(soup: BeautifulSoup, selectors) -> list:
    pairs = []
    for sel in selectors:
        for block in soup.select(sel):
            for row in block.select("tr"):
                cells = row.find_all(["th", "td"], recursive=False) or row.find_all(["th", "td"])
                if len(cells) >= 2:
                    k = cells[0].get_text(" ", strip=True)
                    v = cells[-1].get_text(" ", strip=True)
                    pairs.append((k, v))

            dts = block.find_all("dt")
            dds = block.find_all("dd")
            if dts and dds:
                for dt, dd in zip(dts, dds):
                    pairs.append((dt.get_text(" ", strip=True), dd.get_text(" ", strip=True)))

            for li in block.select("li"):
                text = li.get_text(" ", strip=True)
                if ":" in text:
                    k, v = text.split(":", 1)
                    pairs.append((k, v))
                    continue
                spans = li.find_all("span")
                if len(spans) >= 2:
                    k = spans[0].get_text(" ", strip=True)
                    v = spans[-1].get_text(" ", strip=True)
                    pairs.append((k, v))
    return pairs


def _collect_ld_props(node, out_pairs: list) -> None:
    if isinstance(node, dict):
        ap = node.get("additionalProperty")
        if isinstance(ap, list):
            for p in ap:
                if isinstance(p, dict):
                    out_pairs.append((p.get("name") or p.get("key"), p.get("value")))
        elif isinstance(ap, dict):
            out_pairs.append((ap.get("name") or ap.get("key"), ap.get("value")))

        for k, v in node.items():
            if k in {"@context", "@type", "url", "image", "name", "offers", "description"}:
                continue
            if isinstance(v, (dict, list)):
                _collect_ld_props(v, out_pairs)
    elif isinstance(node, list):
        for item in node:
            _collect_ld_props(item, out_pairs)


def _extract_specs_from_json_ld(soup: BeautifulSoup) -> list:
    pairs = []
    for sc in soup.select('script[type="application/ld+json"]'):
        raw = (sc.string or sc.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        _collect_ld_props(data, pairs)
    return pairs


AMZN_SPEC_SELS = [
    "#productDetails_techSpec_section_1",
    "#productDetails_techSpec_section_2",
    "#technicalSpecifications_section_1",
    "#productDetails_detailBullets_sections1",
    "#detailBullets_feature_div",
    "#poExpander table",
    "#productOverview_feature_div table",
    "table.a-normal.a-spacing-micro",
]

FK_SPEC_SELS = [
    "table._0ZhAN9",
    "table._14cfVK",
    "div._3k-BhJ table",
    "div._1UhVsV table",
    "div.X3BRps table",
    "div.GNDEQ- table",
]


def _extract_amazon_specs(soup: BeautifulSoup) -> dict:
    pairs = []
    pairs.extend(_extract_specs_from_tables(soup, AMZN_SPEC_SELS))
    pairs.extend(_extract_specs_from_json_ld(soup))
    return _normalize_specs(pairs)


def _extract_flipkart_specs(soup: BeautifulSoup) -> dict:
    pairs = []
    pairs.extend(_extract_specs_from_tables(soup, FK_SPEC_SELS))
    pairs.extend(_extract_specs_from_json_ld(soup))
    return _normalize_specs(pairs)


def _extract_specs_from_structured(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    pairs = []
    candidate_keys = [
        "specifications", "specs", "technical_details", "product_details",
        "attributes", "details", "feature_bullets", "about_this_item",
    ]

    def collect(node):
        if isinstance(node, dict):
            if "name" in node and "value" in node and not isinstance(node.get("value"), (dict, list)):
                pairs.append((node.get("name"), node.get("value")))
            for k, v in node.items():
                if k in {"@context", "@type", "url", "image", "description"}:
                    continue
                if isinstance(v, (dict, list)):
                    collect(v)
                elif isinstance(v, (str, int, float)) and k not in {"name", "value"}:
                    pairs.append((k.replace("_", " "), v))
        elif isinstance(node, list):
            for item in node:
                collect(item)

    for key in candidate_keys:
        if key in data:
            collect(data.get(key))

    if not pairs:
        for key in [
            "brand", "model", "color", "size", "material", "item_weight",
            "memory_storage_capacity", "screen_size", "operating_system",
            "ram_memory_installed_size",
        ]:
            val = data.get(key)
            if isinstance(val, (str, int, float)) and str(val).strip():
                pairs.append((key.replace("_", " "), val))

    return _normalize_specs(pairs)


# =============================================================================
# Amazon
# =============================================================================

AMZN_TITLE_SELS = [
    "#productTitle", "span#productTitle", "h1.a-size-large",
    "#title span", "h1 span",
]
AMZN_PRICE_SELS = [
    "span.a-price.priceToPay span.a-offscreen",
    ".priceToPay span.a-offscreen",
    "#corePriceDisplay_desktop_feature_div span.a-offscreen",
    "#corePriceDisplay_desktop_feature_div .a-price-whole",
    ".a-price .a-offscreen",
    "#priceblock_ourprice", "#priceblock_dealprice",
    "#priceblock_saleprice", ".a-price-whole",
    "span[data-a-color='price'] .a-offscreen",
    "#apex_offerDisplay_desktop span.a-offscreen",
    ".reinventPricePriceToPayMargin span.a-offscreen",
]
AMZN_IMG_SELS = [
    "#landingImage", "#imgTagWrapperId img",
    "#imgBlkFront", "img#main-image",
    "#imageBlock img", ".imgTagWrapper img",
]

def _parse_amazon_structured(data: dict) -> Optional[dict]:
    """Convert ScraperAPI structured JSON to our format."""
    if not data: return None
    title = (data.get("name") or data.get("product_title") or "").strip() or None
    # Price comes in various shapes
    price = None
    for key in ["price", "pricing", "list_price", "sale_price", "original_price"]:
        v = data.get(key)
        if v:
            price = _price(str(v))
            if price: break
    # Nested pricing object
    if not price and isinstance(data.get("pricing"), dict):
        for k in ["current_price", "price", "sale_price"]:
            v = data["pricing"].get(k)
            if v:
                price = _price(str(v))
                if price: break

    # Image
    img = None
    imgs = data.get("images") or data.get("product_photos") or []
    if isinstance(imgs, list) and imgs:
        img = imgs[0] if isinstance(imgs[0], str) else None
    if not img:
        img = data.get("main_image") or data.get("image") or data.get("thumbnail")
    img = _norm_img(img)

    url = data.get("url") or data.get("product_url") or ""
    avail = data.get("availability") or "In Stock"
    specs = _extract_specs_from_structured(data)

    if not (title or price or img): return None
    return {"platform": "Amazon", "title": title, "price": price,
            "image_url": img, "url": url, "availability": avail, "specs": specs}

def scrape_amazon(url: str) -> Optional[dict]:
    print(f"\n[Amazon] {url[:80]}")

    # Strategy 1: structured JSON endpoint (cleanest, fastest)
    struct = _scraperapi_structured(url)
    if struct:
        result = _parse_amazon_structured(struct)
        if result and (result.get("title") or result.get("price")):
            print(f"  [Amazon structured] title={repr(str(result.get('title',''))[:50])} price={result.get('price')}")
            return result

    # Strategy 2: ScraperAPI HTML (no render – fast)
    resp = _scraperapi_html(url, render=False)

    # Strategy 3: direct fetch fallback
    if not resp or _blocked(resp.text):
        resp = _direct_fetch(url, "https://www.amazon.in/")

    if not resp or _blocked(resp.text):
        print("  [Amazon] all strategies failed/blocked")
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = None
    for sel in AMZN_TITLE_SELS:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t and len(t) > 3: title = t; break

    # Price from HTML selectors
    price = None
    for sel in AMZN_PRICE_SELS:
        el = soup.select_one(sel)
        if el:
            p = _price(el.get("content") if el.has_attr("content") else el.get_text())
            if p: price = p; break

    # Price from inline scripts (catches dynamic pricing)
    if not price:
        price = _extract_price_from_scripts(html)

    # Image
    img = None
    for sel in AMZN_IMG_SELS:
        el = soup.select_one(sel)
        if el:
            dyn = el.get("data-a-dynamic-image") or ""
            if dyn:
                try:
                    imgs = json.loads(dyn)
                    if isinstance(imgs, dict) and imgs: img = next(iter(imgs)); break
                except Exception: pass
            img = el.get("src") or el.get("data-old-hires") or el.get("data-src")
            if img and img.startswith("http"): break

    # Image from scripts
    if not img:
        img = _extract_image_from_scripts(html)

    # JSON-LD + og fallbacks
    ld_t, ld_p, ld_i = _json_ld(soup)
    og_t, og_p, og_i = _og(soup)
    title = title or ld_t or og_t
    price = price or ld_p or og_p
    img   = _norm_img(img or ld_i or og_i)
    specs = _extract_amazon_specs(soup)

    avail = "In Stock"
    el = soup.select_one("#availability span")
    if el: avail = el.get_text(" ", strip=True) or "In Stock"

    print(f"  [Amazon HTML] title={repr(str(title or '')[:50])} price={price}")
    if not (title or price or img): return None
    return {"platform": "Amazon", "title": title, "price": price,
            "image_url": img, "url": url, "availability": avail, "specs": specs}


# =============================================================================
# Flipkart
# =============================================================================

FK_TITLE_SELS = [
    "span.VU-ZEz", "h1.yhB1nd", ".B_NuCI", "span.B_NuCI",
    "span[class*='VU-ZEz']", "div[class*='GNDEQ-'] h1",
    "div.col.col-7-12 h1", "h1",
]
FK_PRICE_SELS = [
    "div.Nx9bqj.CxhGGd", "div.Nx9bqj", "div[class*='Nx9bqj']",
    "._30jeq3._16Jk6d", "._30jeq3", "div._16Jk6d",
    "[itemprop='price']", "div.UOCQB1", "div._3tbKJL",
    "div.CEmiEU div.Nx9bqj",
]
FK_IMG_SELS = [
    "img.DByuf4", "img._396cs4", "img._2r_T1I", "img._53J4C-",
    "img[src*='rukminim']", "img[src*='flixcart']",
    "div._2r_T1I img", "div._3kidU img",
]

def scrape_flipkart(url: str) -> Optional[dict]:
    print(f"\n[Flipkart] {url[:80]}")

    host = urlparse(url).netloc.lower()
    is_short_fk = host in {"dl.flipkart.com", "fkrt.cc", "fkrt.it", "fkrt.to"}

    # Strategy 1: ScraperAPI HTML.
    # For short links: fast non-render probe first, then rendered fallback.
    if is_short_fk:
        resp = _scraperapi_html_flipkart_short(url, timeout=15)
        if not resp or _blocked(resp.text):
            resp = _scraperapi_html(url, render=False, timeout=20)
    else:
        resp = _scraperapi_html(url, render=False, timeout=25)

    # Strategy 2: direct fetch
    if not resp or _blocked(resp.text):
        resp = _direct_fetch(url, "https://www.flipkart.com/", fast=True)

    # Strategy 3: ScraperAPI render fallback for hard blocks/captcha.
    if (not resp or _blocked(resp.text)) and SCRAPER_API_KEY and not is_short_fk:
        resp = _scraperapi_html(url, render=True, timeout=25)

    if not resp or _blocked(resp.text):
        print("  [Flipkart] all strategies failed/blocked")
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Use canonical URL when available.
    canon = soup.select_one('link[rel="canonical"]')
    if canon:
        href = (canon.get("href") or "").strip()
        if href:
            url = urljoin(url, href)

    # Title
    title = None
    for sel in FK_TITLE_SELS:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t and len(t) > 3: title = t; break

    # Price
    price = None
    for sel in FK_PRICE_SELS:
        el = soup.select_one(sel)
        if el:
            p = _price(el.get("content") if el.has_attr("content") else el.get_text())
            if p: price = p; break

    if not price:
        price = _extract_price_from_scripts(html)

    # When pid is present, prefer price tied to that pid (reduces random embedded-price mismatches).
    pid_price = _extract_flipkart_pid_price(html, url)
    if pid_price and (not price or (price < 1000 and pid_price >= 1000)):
        price = pid_price

    # Image
    img = None
    for sel in FK_IMG_SELS:
        el = soup.select_one(sel)
        if el:
            src = el.get("src") or el.get("data-src") or ""
            if src and src.startswith("http"):
                img = src.split(",")[0].split()[0]; break

    if not img:
        img = _extract_image_from_scripts(html)

    # JSON-LD + og fallbacks
    ld_t, ld_p, ld_i = _json_ld(soup)
    og_t, og_p, og_i = _og(soup)
    title = title or ld_t or og_t
    price = price or ld_p or og_p
    img   = _norm_img(img or ld_i or og_i)
    specs = _extract_flipkart_specs(soup)

    print(f"  [Flipkart HTML] title={repr(str(title or '')[:50])} price={price}")
    if not (title or price or img):
        # Last resort: if this was a short URL, resolve it and retry once with direct product URL.
        if is_short_fk:
            resolved = resolve_url(url)
            if resolved and resolved != url:
                print(f"  [Flipkart] retrying via resolved URL -> {resolved}")
                return scrape_flipkart(resolved)
        return None
    return {"platform": "Flipkart", "title": title, "price": price,
            "image_url": img, "url": url, "availability": "In Stock", "specs": specs}


# =============================================================================
# Cross-platform search
# =============================================================================

# def _clean_q(name: str) -> str:
#     name = re.sub(r'\([^)]*\)', '', name)
#     return " ".join(name.split()[:7]).strip()

# def search_amazon(product_name: str) -> Optional[dict]:
#     q = _clean_q(product_name)
#     print(f"  [Amazon Search] '{q}'")
#     resp = _scraperapi_html(f"https://www.amazon.in/s?k={quote_plus(q)}", render=False)
#     if not resp: resp = _direct_fetch(f"https://www.amazon.in/s?k={quote_plus(q)}", "https://www.amazon.in/")
#     if not resp: return None
#     soup = BeautifulSoup(resp.text, "html.parser")
#     for card in soup.select('[data-component-type="s-search-result"]'):
#         link = card.select_one('h2 a.a-link-normal[href], h2 a[href]')
#         if not link: continue
#         href = link.get("href", "")
#         if not href or "/s?" in href: continue
#         data = scrape_amazon(urljoin("https://www.amazon.in", href))
#         if data: data["availability"] = "Found in search"; return data
#     return None

# def search_flipkart(product_name: str) -> Optional[dict]:
#     q = _clean_q(product_name)
#     print(f"  [Flipkart Search] '{q}'")
#     resp = _scraperapi_html(f"https://www.flipkart.com/search?q={quote_plus(q)}", render=False)
#     if not resp: resp = _direct_fetch(f"https://www.flipkart.com/search?q={quote_plus(q)}", "https://www.flipkart.com/")
#     if not resp: return None
#     soup = BeautifulSoup(resp.text, "html.parser")
#     for a in soup.select('a[href*="/p/"], a[href*="pid="]'):
#         href = a.get("href", "")
#         if not href: continue
#         data = scrape_flipkart(urljoin("https://www.flipkart.com", href))
#         if data: data["availability"] = "Found in search"; return data
#     return None


# =============================================================================
# Name from URL
# =============================================================================

def _name_from_url(url: str) -> Optional[str]:
    try:
        path = unquote(urlparse(url).path)
        host = urlparse(url).netloc.lower()
        if "amazon" in host:
            m = re.search(r'/([^/]+)/dp/', path)
            if m: return re.sub(r'[-_]+', ' ', m.group(1)).strip()
        if "flipkart" in host:
            m = re.search(r'/([^/]+)/p/', path)
            if m: return re.sub(r'[-_]+', ' ', m.group(1)).strip()
        segs = [s for s in path.split('/') if '-' in s and len(s) > 10]
        if segs: return re.sub(r'[-_]+', ' ', max(segs, key=len)).strip()
    except Exception: pass
    return None


# =============================================================================
# Main entry point
# =============================================================================

def analyze_url(raw_url: str) -> dict:
    url = raw_url.strip()
    if not url:
        return {"success": False, "error": "URL is required"}

    print(f"\n{'='*65}")
    print(f"[ANALYZE] {url}")
    print(f"[ANALYZE] ScraperAPI: {'active ✅' if SCRAPER_API_KEY else 'MISSING ❌'}")

    host = urlparse(url).netloc.lower().lstrip("www.")
    is_fk_short = host in {"dl.flipkart.com", "fkrt.cc", "fkrt.it", "fkrt.to"}

    # Resolve short links. For Flipkart short links, defer to scrape_flipkart()
    # to avoid paying the same resolve cost twice.
    if (not is_fk_short) and (_is_short(url) or ("amazon.in" not in url and "flipkart.com" not in url)):
        url = resolve_url(url)
        print(f"[ANALYZE] Resolved: {url}")

    # Detect platform
    url_l = url.lower()
    if "amazon.in" in url_l or "amazon.com" in url_l:
        source = "amazon"
    elif "flipkart.com" in url_l:
        source = "flipkart"
    else:
        return {"success": False,
                "error": "Could not identify platform. Please paste a direct Amazon.in or Flipkart.com product URL."}

    print(f"[ANALYZE] Platform: {source}")

    results: dict = {}
    product_name: Optional[str] = None

    # Scrape source platform
    if source == "amazon":
        data = scrape_amazon(url)
        if data: results["amazon"] = data; product_name = data.get("title")
    else:
        data = scrape_flipkart(url)
        if data: results["flipkart"] = data; product_name = data.get("title")

    # Name fallback from URL
    if not product_name:
        product_name = _name_from_url(url)
        print(f"[ANALYZE] URL-derived name: {product_name}")

    if not results and not product_name:
        return {
            "success": False,
            "error": (
                "Could not extract product details from this URL. "
                f"(ScraperAPI key: {'active' if SCRAPER_API_KEY else 'MISSING'}). "
                "Please try a full product URL, not a short link."
            )
        }

    if not results and product_name:
        results[source] = {
            "platform": "Amazon" if source == "amazon" else "Flipkart",
            "title": product_name, "price": None, "image_url": None,
            "url": url, "availability": "Price could not be verified", "specs": {},
        }

    # # Search other platform
    # if product_name:
    #     if source == "amazon" and "flipkart" not in results:
    #         fk = search_flipkart(product_name)
    #         if fk: results["flipkart"] = fk
    #     elif source == "flipkart" and "amazon" not in results:
    #         az = search_amazon(product_name)
    #         if az: results["amazon"] = az

    # Build comparison
    cheapest_platform = None
    cheapest_price = float("inf")
    for plat, info in results.items():
        p = info.get("price")
        if p and p < cheapest_price: cheapest_price = p; cheapest_platform = plat

    prices = [v.get("price") for v in results.values() if v.get("price")]
    price_diff = abs(prices[0] - prices[1]) if len(prices) == 2 else None

    src_data = results.get(source) or (next(iter(results.values())) if results else None)
    current_price = src_data.get("price") if src_data else None

    future_prices: dict = {}
    recommendation = "PRICE_UNAVAILABLE"
    max_savings = None
    if current_price:
        future_prices = {
            "7_days":  round(current_price * 0.98, 2),
            "15_days": round(current_price * 0.95, 2),
            "30_days": round(current_price * 0.92, 2),
            "60_days": round(current_price * 0.90, 2),
            "90_days": round(current_price * 0.88, 2),
        }
        max_savings = round(current_price - min(future_prices.values()), 2)
        recommendation = "WAIT" if (max_savings / current_price) > 0.05 else "BUY_NOW"

    print(f"[ANALYZE] Done. platforms={list(results.keys())} price={current_price}")

    return {
        "success": True,
        "product_name": product_name,
        "source_platform": source,
        "source_url": raw_url,
        "resolved_url": url,
        "platforms_found": list(results.keys()),
        "product": {
            "name": product_name,
            "current_price": current_price,
            "source": source,
            "url": url,
            "image_url": src_data.get("image_url") if src_data else None,
        },
        "prediction": {
            "future_prices": future_prices,
            "recommendation": recommendation,
            "max_savings": max_savings,
            "best_time_days": 90 if recommendation == "WAIT" else None,
            "confidence": 0.85 if len(results) == 2 else 0.60,
        },
        "amazon":   results.get("amazon",   {"found": False, "message": "Not found on Amazon"}),
        "flipkart": results.get("flipkart", {"found": False, "message": "Not found on Flipkart"}),
        "comparison": {
            "cheapest_platform": cheapest_platform,
            "cheapest_price": cheapest_price if cheapest_price != float("inf") else None,
            "price_difference": price_diff,
            "savings_platform": cheapest_platform,
            "both_found": len(results) == 2,
        },
    }
