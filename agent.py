"""LangGraph 버전 추천 에이전트

흐름: 프로필 불러오기 → 질문 분석 → 후보 검색 → (후보가 부족하면) 조건 완화 후 재검색 → 추천 답변 생성
                     ↘ (플레이 기록을 못 불러오면) Steam ID 다시 요청
"""
import json
import os
import re
import time
from typing import TypedDict

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from langgraph.graph import END, START, StateGraph

from steam_api import get_owned_games, get_prices, get_reviews_many, resolve_steamid

load_dotenv()

PLAYED_MIN = 600   # 이만큼(분) 이상 플레이한 게임은 추천에서 제외 (10시간)
MIN_CANDS = 6      # 후보가 이보다 적으면 조건을 완화해서 다시 검색
MAX_RELAX = 2      # 조건 완화는 최대 2번까지
DEFAULT_COUNT = 5  # 기본 추천 개수
STORE_URL = "https://store.steampowered.com/app/{appid}/"                   # 스팀 상점 페이지
IMAGE_URL = "https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"  # 대표 이미지
MAX_COUNT = 10     # 최대 추천 개수
REVIEW_TOP = 10    # 리뷰를 가져올 후보 수 (검색 순위 상위)

llm = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)),
)
_db = chromadb.PersistentClient(path="chroma_db")
try:
    col = _db.get_collection("games")
    if col.count() == 0:
        raise ValueError("empty")
except Exception:
    # 배포 서버처럼 벡터 DB가 없으면 games.csv로 자동 생성
    from build_db import build
    col = build(_db)


# =====================================================================
# Gemini 호출 도우미
# =====================================================================
def list_models() -> tuple[list[str], list[str]]:
    """내 키로 쓸 수 있는 Flash / Flash-Lite 모델을 최신순으로 정리"""
    skip = ("preview", "exp", "image", "tts", "live", "audio", "thinking")
    found = []
    for m in llm.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions or "flash" not in name:
            continue
        if any(s in name for s in skip):
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        if ver:
            found.append((-float(ver.group(1)), name))
    names = list(dict.fromkeys(n for _, n in sorted(found)))  # 중복 제거
    flash = [n for n in names if "lite" not in n][:3]
    lite = [n for n in names if "lite" in n][:2]
    pref = os.getenv("GEMINI_MODEL")
    if pref:
        flash = [pref] + [n for n in flash if n != pref]
    if not flash and not lite:
        raise SystemExit("쓸 수 있는 Gemini 모델을 못 찾았어요. GEMINI_API_KEY를 확인하세요.")
    return flash, lite


FLASH, LITE = list_models()
print(f"답변용 모델: {', '.join(FLASH)}")
print(f"분석용 모델: {', '.join(LITE) or '(Lite 없음, 답변용 모델 사용)'}")


def ask_llm(prompt: str, json_mode: bool = False, light: bool = False) -> str:
    """light=True면 가벼운 Lite 모델 먼저. 모델이 바쁘면(503/429) 다음 모델로 넘어감"""
    models = (LITE + FLASH) if light else (FLASH + LITE)
    config = types.GenerateContentConfig(
        temperature=0.2 if json_mode else 0.7,
        response_mime_type="application/json" if json_mode else None,
    )
    last_err = None
    for model in models:
        try:
            start = time.time()
            resp = llm.models.generate_content(model=model, contents=prompt, config=config)
            print(f"      ({model} {time.time() - start:.1f}초)")
            return resp.text
        except errors.APIError as e:
            if e.code in (404, 429, 500, 503):
                print(f"      ({model} 사용 불가 {e.code} → 다음 모델)")
                last_err = e
                continue
            raise
    raise last_err


# 답변을 한 조각씩 받아볼 함수 (웹 화면에서 설정, 터미널에서는 None)
on_token = None


def ask_llm_stream(prompt: str, callback) -> str:
    """답변을 조각조각 받아서 callback에 넘겨줌 (화면에 바로바로 표시용)"""
    config = types.GenerateContentConfig(temperature=0.7)
    last_err = None
    for model in FLASH + LITE:
        parts = []
        try:
            start = time.time()
            for chunk in llm.models.generate_content_stream(model=model, contents=prompt, config=config):
                if chunk.text:
                    parts.append(chunk.text)
                    callback(chunk.text)
            print(f"      ({model} 스트리밍 {time.time() - start:.1f}초)")
            return "".join(parts)
        except errors.APIError as e:
            if e.code in (404, 429, 500, 503) and not parts:
                print(f"      ({model} 사용 불가 {e.code} → 다음 모델)")
                last_err = e
                continue
            raise
    raise last_err


# =====================================================================
# 그래프 상태: 노드들이 주고받는 데이터
# =====================================================================
class State(TypedDict, total=False):
    question: str        # 사용자 질문
    steamid: str | None  # Steam ID
    profile: dict        # 플레이 기록 요약
    plan: dict           # 질문 분석 결과 (검색어, 조건)
    filters: dict        # 현재 검색 조건 (완화되면 바뀜)
    relax_count: int     # 조건 완화 횟수
    candidates: list     # 검색된 후보 게임
    answer: str          # 최종 답변
    history: list        # 이전 대화 [{"q": 질문, "recs": [추천한 게임]}] (최근 몇 개)
    exclude: list        # 이미 추천한 게임 이름 (다시 추천하지 않음)
    count: int           # 추천 개수
    friend: dict         # 같이 할 친구의 플레이 기록 (없으면 혼자 추천)
    likes: list          # 좋아요 누른 게임 [{"name", "genres"}]
    dislikes: list       # 별로예요 누른 게임 [{"name", "genres"}]


# =====================================================================
# 노드
# =====================================================================
def make_profile(steamid: str) -> dict | None:
    """Steam ID → 플레이 기록 요약 (나와 친구 모두 이 함수로 불러옴)"""
    games = get_owned_games(steamid)
    if not games:
        return None
    return {
        "count": len(games),
        "top": [f"{g['name']} ({g['playtime_forever'] / 60:.0f}h)" for g in games[:10]],
        "recent": [f"{g['name']} ({g['playtime_2weeks'] / 60:.1f}h)"
                   for g in games if g.get("playtime_2weeks")],
        "owned": {g["appid"]: g for g in games},
    }


def load_profile(state: State) -> dict:
    """[노드 1] Steam API로 플레이 기록 불러오기 (이미 있으면 건너뜀)"""
    if state.get("profile") or not state.get("steamid"):
        return {}
    prof = make_profile(state["steamid"])
    return {"profile": prof} if prof else {}


def names_of(items: list) -> str:
    return ", ".join(x["name"] for x in items) if items else ""


def review_text(c: dict) -> str:
    """프롬프트에 넣을 리뷰 요약 (없으면 빈 문자열)"""
    rv = c.get("reviews")
    if not rv:
        return ""
    head = f"\n    [리뷰] {rv['lang']} 평가 {rv['summary']} ({rv['total']:,}개)"
    lines = [f"\n      - ({'추천' if it['up'] else '비추천'}, {it['hours']}시간 플레이) {it['text']}"
             for it in rv["items"]]
    return head + "".join(lines)


def price_text(c: dict) -> str:
    """후보 게임의 현재 가격 문구"""
    if c.get("free"):
        return "무료"
    if not c.get("price_now"):
        return "가격 정보 없음"
    if c.get("discount"):
        return f"{c['price_now']} ({c['discount']}% 할인 중, 정가 {c['price_orig']})"
    return c["price_now"]


def ask_steam_id(state: State) -> dict:
    """[노드 1-1] 플레이 기록이 없으면 다시 요청"""
    return {"answer": "플레이 기록을 불러오지 못했어요. Steam ID가 맞는지, "
                      "프로필의 '게임 세부 정보'가 공개로 되어 있는지 확인해 주세요."}


def analyze(state: State) -> dict:
    """[노드 2] 한국어 질문 → 영어 검색어(1~3개) + 조건"""
    p = state["profile"]
    friend = state.get("friend")
    hist = state.get("history") or []
    prev = "\n".join(f'- "{h["q"]}" → recommended: {", ".join(h["recs"]) or "none"}' for h in hist[-3:])
    extra = ""
    if friend:
        extra += (f"\nThe user will play TOGETHER with a friend. Friend's most played games: "
                  f"{', '.join(friend['top'])}\nWrite queries for multiplayer/co-op games both would enjoy.")
    if state.get("likes"):
        extra += f"\nGames the user LIKED before (find similar ones): {names_of(state['likes'])}"
    if state.get("dislikes"):
        extra += f"\nGames the user DISLIKED (avoid similar ones): {names_of(state['dislikes'])}"
    prompt = f"""You turn a Korean game-recommendation request into a search plan.
User's most played games: {', '.join(p['top'])}{extra}
Previous requests in this conversation (oldest first):
{prev or '- (none)'}
New request: "{state['question']}"

If the new request is a follow-up such as "다른 거", "또 추천해줘", "비슷한 거 더",
keep the conditions (co-op, solo, owned, genre) of the previous request and look for different games.

Return JSON with these keys:
- "queries": a list of 1-3 short English search queries describing the kind of game wanted
  (genre, mood, mechanics). If the request is specific, use one query.
  If it is vague, write one query per distinct taste you see in the user's most played games
  (for example "co-op survival crafting sandbox" and "tactical team shooter").
- "coop": true if they want to play together with friends, otherwise null
- "single": true if they want to play alone, otherwise null
- "owned_only": true only if they want games they already own, otherwise false
- "count": the number of games they explicitly asked for (e.g. "10개" → 10), otherwise null
- "sale_only": true if they want games that are currently on sale/discounted (세일, 할인), otherwise false"""
    plan = json.loads(ask_llm(prompt, json_mode=True, light=True))
    if isinstance(plan.get("queries"), str):
        plan["queries"] = [plan["queries"]]
    filters = {k: True for k in ("coop", "single", "owned_only", "sale_only") if plan.get(k)}
    if friend and not plan.get("single"):
        filters["together"] = True  # 친구와 같이 → 멀티플레이 또는 협동 게임만
    try:  # 질문에 숫자가 있으면 그 숫자, 없으면 화면에서 고른 개수
        count = int(plan.get("count") or state.get("count") or DEFAULT_COUNT)
    except (TypeError, ValueError):
        count = state.get("count") or DEFAULT_COUNT
    count = max(1, min(count, MAX_COUNT))
    return {"plan": plan, "filters": filters, "relax_count": 0, "count": count}


def build_where(filters: dict, profile: dict) -> dict | None:
    conds = [{k: True} for k in ("coop", "single") if filters.get(k)]
    if filters.get("together"):
        conds.append({"$or": [{"coop": True}, {"multi": True}]})
    if filters.get("owned_only"):
        conds.append({"appid": {"$in": [int(a) for a in profile["owned"]]}})
    if len(conds) > 1:
        return {"$and": conds}
    return conds[0] if conds else None


def retrieve(state: State) -> dict:
    """[노드 3] 검색어마다 벡터 DB 검색 → 합치기 → 이미 많이 한 게임 제외"""
    profile = state["profile"]
    friend = state.get("friend")
    sale_only = state["filters"].get("sale_only")
    exclude = set(state.get("exclude") or []) | {d["name"] for d in state.get("dislikes") or []}
    queries = state["plan"].get("queries") or [state["question"]]
    where = build_where(state["filters"], profile)
    count = state.get("count") or DEFAULT_COUNT
    want = max(15, count * 3)  # 추천 개수의 3배만큼 후보 확보
    if sale_only:
        want *= 3  # 세일 중인 게임만 고르려면 더 많이 찾아봐야 함
    n_each = max(6, want // len(queries) + 2) + min(len(exclude), 12)  # 제외할 게임만큼 더 검색

    seen, cands = set(), []
    for q in queries:
        res = col.query(query_texts=[q], n_results=n_each, where=where)
        for m, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
            if m["appid"] in seen or m["name"] in exclude:
                continue
            seen.add(m["appid"])
            mine = profile["owned"].get(m["appid"])
            minutes = mine["playtime_forever"] if mine else 0
            if minutes >= PLAYED_MIN:
                continue
            fr = friend["owned"].get(m["appid"]) if friend else None
            cands.append({
                "name": m["name"], "genres": m["genres"], "modes": m["categories"],
                "positive": m["positive_ratio"], "median_hours": m["median_playtime_hours"],
                "owned": f"보유 중 ({minutes / 60:.0f}시간 플레이)" if mine else "미보유",
                "is_owned": bool(mine), "price": m.get("price", 0),
                "friend_owned": bool(fr),
                "friend_hours": round(fr["playtime_forever"] / 60) if fr else 0,
                "appid": m["appid"],
                "store_url": STORE_URL.format(appid=m["appid"]),
                "image": IMAGE_URL.format(appid=m["appid"]),
                "query": q, "dist": dist, "summary": doc[:400],
            })
    cands.sort(key=lambda c: c["dist"])
    cands = cands[:want]

    # 실시간 가격·할인 (스팀 상점, 원화 기준)
    prices = get_prices([c["appid"] for c in cands])
    for c in cands:
        pr = prices.get(c["appid"])
        if pr:
            c.update(free=pr["free"], price_now=pr["now"], price_orig=pr["orig"], discount=pr["discount"])
    if sale_only:
        cands = [c for c in cands if c.get("discount")]
    cands = cands[:max(15, count * 3)]

    # 실제 플레이어 리뷰 (상위 후보만, 병렬로 가져옴)
    top = cands[:max(REVIEW_TOP, count + 3)]
    reviews = get_reviews_many([c["appid"] for c in top])
    for c in top:
        if reviews.get(c["appid"]):
            c["reviews"] = reviews[c["appid"]]
    print(f"      후보 {len(cands)}개 (조건: {state['filters'] or '없음'}, "
          f"가격 {len(prices)}개 · 리뷰 {len(reviews)}개 조회)")
    return {"candidates": cands}


def relax(state: State) -> dict:
    """[노드 3-1] 후보가 부족할 때 조건 완화: 1차는 협동/싱글/세일 조건, 2차는 전부 해제"""
    f = dict(state["filters"])
    count = state.get("relax_count", 0) + 1
    if count == 1 and any(k in f for k in ("coop", "single", "sale_only")):
        for k in ("coop", "single", "sale_only"):
            f.pop(k, None)
    else:
        f = {}
    print(f"      조건 완화 {count}차 → {f or '조건 없음'}")
    return {"filters": f, "relax_count": count}


def generate(state: State) -> dict:
    """[노드 4] 플레이 기록 + 후보를 프롬프트에 넣어 추천 답변 생성 (RAG)"""
    p = state["profile"]
    context = "\n".join(
        f"[{i + 1}] {c['name']} | 장르: {c['genres']} | 방식: {c['modes']} | "
        f"긍정 {c['positive']:.0%} | 보통 총 플레이 {c['median_hours']}시간 | {c['owned']} | "
        + (f"친구: {'보유 (' + str(c['friend_hours']) + '시간)' if c['friend_owned'] else '미보유'} | "
           if state.get("friend") else "")
        + f"가격: {price_text(c)} | 상점: {c['store_url']}\n"
        f"    {c['summary']}"
        + review_text(c)
        for i, c in enumerate(state["candidates"])
    )
    count = state.get("count") or DEFAULT_COUNT
    length = "2~3문장" if count <= 5 else "1~2문장"
    hist = state.get("history") or []
    prev_txt = "\n".join(f"- 질문: {h['q']} / 추천했던 게임: {', '.join(h['recs']) or '없음'}"
                         for h in hist[-3:]) or "없음"
    friend = state.get("friend")
    friend_txt = (f"\n[같이 할 친구]\n- 보유 게임 {friend['count']}개\n"
                  f"- 많이 한 게임: {', '.join(friend['top'])}\n") if friend else ""
    feedback_txt = ""
    if state.get("likes"):
        feedback_txt += f"\n- 좋아요 누른 게임: {names_of(state['likes'])} (이런 게임과 비슷한 걸 더 좋아함)"
    if state.get("dislikes"):
        feedback_txt += f"\n- 별로예요 누른 게임: {names_of(state['dislikes'])} (이런 느낌은 피하기)"
    friend_rule = ("\n- 친구와 같이 할 게임을 추천하는 거야. 나와 친구의 플레이 기록을 모두 연결해서 설명하고, "
                   "둘 다 가진 게임이면 '둘 다 갖고 있어서 바로 같이 시작할 수 있다'고 먼저 추천해.") if friend else ""
    relaxed = ("\n(참고: 조건에 딱 맞는 후보가 부족해서 조건을 조금 넓혀 찾은 후보도 있어. "
               "그런 게임은 어떤 점이 조건과 다른지 짧게 알려줘.)") if state.get("relax_count") else ""
    prompt = f"""너는 스팀 게임 추천 도우미야. 아래 [사용자 정보]와 [후보 게임]만 근거로 답해.
후보 목록에 없는 게임은 절대 추천하지 마.{relaxed}

[사용자 정보]
- 보유 게임 {p['count']}개
- 많이 한 게임: {', '.join(p['top'])}
- 최근 2주 플레이: {', '.join(p['recent']) or '없음'}
{friend_txt}{feedback_txt}
[이전 대화]
{prev_txt}

[질문]
{state['question']}

[후보 게임]
{context}

[답변 규칙]
- 질문 조건에 가장 잘 맞는 게임 {count}개를 골라, 잘 맞는 순서대로 번호를 붙여 추천해.
  조건에 맞는 후보가 {count}개보다 적으면 억지로 채우지 말고 맞는 것만 추천해.
- 이전 대화에서 이미 추천했던 게임은 다시 추천하지 마.
- 각 게임마다 왜 이 사용자에게 맞는지 플레이 기록과 연결해서 {length}으로 설명해.
- 보유했지만 거의 안 한 게임이 조건에 맞으면 먼저 추천하고 "이미 갖고 있는 게임"이라고 알려줘.
- 추천하는 게임 이름은 반드시 [게임 이름](상점 주소) 형식의 링크로 써. 게임 이름은 후보 목록과 똑같이 써.
- [리뷰]가 있는 게임은 실제 플레이어 리뷰 내용을 짧게 인용하거나 요약해서 근거로 써
  (예: 리뷰에서 "친구랑 하면 시간 순삭"이라는 평이 많아요). 리뷰에 없는 내용을 리뷰라고 지어내지 마.
- 할인 중인 게임은 할인율과 가격을 짧게 알려줘. 그 외 가격은 꼭 필요할 때만 언급해.{friend_rule}
- 마지막 줄에 "근거: 내 플레이 기록 · 스토어 게임 설명 · 플레이어 리뷰"라고 적어.
- 한국어, 친근한 말투로."""
    if on_token:
        return {"answer": ask_llm_stream(prompt, on_token)}
    return {"answer": ask_llm(prompt)}


# =====================================================================
# 분기 조건
# =====================================================================
def route_after_profile(state: State) -> str:
    return "analyze" if state.get("profile") else "ask_steam_id"


def route_after_retrieve(state: State) -> str:
    need = max(MIN_CANDS, (state.get("count") or DEFAULT_COUNT) + 2)
    if len(state["candidates"]) < need and state.get("relax_count", 0) < MAX_RELAX:
        return "relax"
    return "generate"


# =====================================================================
# 그래프 조립
# =====================================================================
def build_graph():
    g = StateGraph(State)
    g.add_node("load_profile", load_profile)
    g.add_node("ask_steam_id", ask_steam_id)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve)
    g.add_node("relax", relax)
    g.add_node("generate", generate)

    g.add_edge(START, "load_profile")
    g.add_conditional_edges("load_profile", route_after_profile,
                            {"analyze": "analyze", "ask_steam_id": "ask_steam_id"})
    g.add_edge("ask_steam_id", END)
    g.add_edge("analyze", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"relax": "relax", "generate": "generate"})
    g.add_edge("relax", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


app = build_graph()

LABELS = {
    "load_profile": "플레이 기록 불러오기", "ask_steam_id": "Steam ID 확인 요청",
    "analyze": "질문 분석", "retrieve": "후보 검색", "relax": "조건 완화",
    "generate": "추천 답변 생성",
}


def ranked_picks(answer: str, candidates: list) -> list[dict]:
    """답변에 실제로 등장한 후보 게임을, 답변에 나온 순서(=추천 순위)대로 정렬"""
    answer = answer or ""
    found = [(answer.find(c["name"]), c) for c in candidates if c["name"] in answer]
    return [c for _, c in sorted(found, key=lambda x: x[0])]


def recommended_names(answer: str, candidates: list) -> list[str]:
    """이번에 추천한 게임 이름 (대화 기억용)"""
    return [c["name"] for c in ranked_picks(answer, candidates)]


def remember(memory: dict, question: str, result: dict):
    """대화 기억 업데이트: 질문과 추천한 게임을 저장"""
    recs = recommended_names(result.get("answer", ""), result.get("candidates", []))
    memory["history"] = (memory.get("history", []) + [{"q": question, "recs": recs}])[-5:]
    memory["exclude"] = list(dict.fromkeys(memory.get("exclude", []) + recs))


def run(question: str, steamid: str | None, profile: dict | None = None,
        memory: dict | None = None) -> dict:
    """그래프를 실행하면서 지나가는 노드를 출력하고, 최종 상태를 돌려줌"""
    memory = memory or {}
    state: State = {"question": question, "steamid": steamid,
                    "history": memory.get("history", []), "exclude": memory.get("exclude", [])}
    if profile:
        state["profile"] = profile
    final = dict(state)
    for chunk in app.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"  [완료] {LABELS.get(node, node)}")
            final.update(update or {})
    return final


if __name__ == "__main__":
    # 발표 자료용 그래프 구조를 Mermaid 형식으로 저장
    with open("graph.mmd", "w", encoding="utf-8") as f:
        f.write(app.get_graph().draw_mermaid())

    user = os.getenv("STEAM_ID") or input("Steam ID 또는 프로필 주소: ")
    steamid = resolve_steamid(user)
    profile = None
    memory = {}  # 대화 기억 (이전 질문, 이미 추천한 게임)
    print("질문을 입력하세요 (빈 줄 입력 시 종료)\n")

    while True:
        q = input("질문> ").strip()
        if not q:
            break
        try:
            result = run(q, steamid, profile, memory)
        except errors.APIError as e:
            print(f"\n[오류] Gemini API {e.code}: {e.message}\n  1분 정도 뒤에 다시 질문하세요.\n")
            continue
        except Exception as e:
            print(f"\n[오류] {type(e).__name__}: {e}\n")
            continue
        profile = result.get("profile", profile)  # 다음 질문부터는 다시 불러오지 않음
        remember(memory, q, result)
        if result.get("plan"):
            print(f"\n[검색어] {result['plan'].get('queries')}")
            print(f"[후보] {', '.join(c['name'] for c in result.get('candidates', []))}")
        print(f"\n{result.get('answer', '')}\n")
