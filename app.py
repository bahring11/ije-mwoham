"""이제뭐함 - Streamlit 웹 화면
실행: streamlit run app.py
"""
import os
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

st.set_page_config(page_title="이제뭐함", page_icon="🎮", layout="wide")

with st.spinner("준비 중이에요... 서버가 처음 켜질 때는 2~3분 걸릴 수 있어요."):
    import agent  # noqa: E402  (모델 목록 조회, 벡터 DB 준비 등 처음 한 번만 실행됨)
from steam_api import resolve_steamid  # noqa: E402

MAX_QUESTIONS = 10  # 한 사람(브라우저 세션)당 질문 횟수 제한

EXAMPLES = [
    "친구랑 같이 할 협동 게임 추천해줘",
    "오늘 1시간 정도 혼자 할 게임",
    "지금 세일 중인 게임 중에 추천해줘",
]

ss = st.session_state
ss.setdefault("profile", None)
ss.setdefault("steamid", None)
ss.setdefault("friend", None)        # 같이 할 친구의 플레이 기록
ss.setdefault("messages", [])
ss.setdefault("auto_tried", False)
ss.setdefault("asked", 0)
ss.setdefault("memory", {})          # 대화 기억 (이전 질문, 이미 추천한 게임)
ss.setdefault("likes", {})           # 좋아요 {게임 이름: 게임 정보}
ss.setdefault("dislikes", {})        # 별로예요 {게임 이름: 게임 정보}
ss.setdefault("wishlist", {})        # 찜 목록 {게임 이름: 게임 정보}


def fetch_profile(user_input: str):
    sid = resolve_steamid(user_input) if user_input.strip() else None
    return sid, (agent.make_profile(sid) if sid else None)


def load(user_input: str):
    with st.spinner("플레이 기록 불러오는 중..."):
        sid, prof = fetch_profile(user_input)
    if prof:
        ss.profile, ss.steamid, ss.messages, ss.memory = prof, sid, [], {}
    else:
        st.error("플레이 기록을 불러오지 못했어요. Steam ID와 프로필 공개 설정을 확인하세요.")


def load_friend(user_input: str):
    with st.spinner("친구 플레이 기록 불러오는 중..."):
        sid, prof = fetch_profile(user_input)
    if prof and sid == ss.steamid:
        st.error("내 Steam ID예요. 친구의 Steam ID를 넣어주세요.")
    elif prof:
        ss.friend, ss.memory = prof, {}
        return True
    else:
        st.error("친구 플레이 기록을 불러오지 못했어요. 친구 프로필의 '게임 세부 정보'가 공개인지 확인하세요.")


# ---------- 카드 버튼 동작 (좋아요 / 별로예요 / 찜) ----------
def _brief(g: dict) -> dict:
    keep = ("name", "genres", "store_url", "image", "appid", "price_now", "price_orig", "discount", "free")
    return {k: g.get(k) for k in keep}


def toggle(bucket: str, g: dict):
    box = ss[bucket]
    if g["name"] in box:
        box.pop(g["name"])
    else:
        box[g["name"]] = _brief(g)
        # 좋아요와 별로예요는 동시에 누를 수 없게
        other = {"likes": "dislikes", "dislikes": "likes"}.get(bucket)
        if other:
            ss[other].pop(g["name"], None)


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("🎮 내 라이브러리")
    default_id = os.getenv("STEAM_ID", "")
    user = st.text_input("Steam ID 또는 프로필 주소", value=default_id)
    with st.expander("Steam ID는 어디서 찾나요?"):
        st.markdown(
            "1. 스팀에서 **내 프로필**을 열어요.\n"
            "2. 빈 곳을 오른쪽 클릭 → **페이지 주소 복사**\n"
            "3. 복사한 주소를 위 칸에 그대로 붙여넣으면 돼요.\n\n"
            "⚠️ 프로필 편집 → 개인정보 설정에서 **'게임 세부 정보'를 공개**로 바꿔야 "
            "플레이 기록을 불러올 수 있어요."
        )
    if st.button("불러오기", use_container_width=True, type="primary"):
        load(user)
    elif default_id and ss.profile is None and not ss.auto_tried:
        ss.auto_tried = True
        load(default_id)

    if ss.profile:
        st.slider("추천 개수", 3, 8, agent.DEFAULT_COUNT, key="count",
                  help="질문에 '10개 추천해줘'처럼 숫자를 쓰면 그 숫자가 우선이에요 (최대 10개).")

        # ----- 친구와 같이
        st.divider()
        st.subheader("👥 친구랑 같이 하기")
        if ss.friend:
            mine = set(ss.profile["owned"])
            both = mine & set(ss.friend["owned"])
            st.success(f"친구 연결됨 · 친구 보유 게임 {ss.friend['count']}개 · 둘 다 가진 게임 {len(both)}개")
            st.caption("지금은 친구와 같이 할 수 있는 멀티·협동 게임 위주로 추천해요.")
            if st.button("친구 연결 해제", use_container_width=True):
                ss.friend, ss.memory = None, {}
                st.rerun()
        else:
            friend_input = st.text_input("친구 Steam ID 또는 프로필 주소", key="friend_input",
                                         placeholder="친구 프로필 주소를 붙여넣으세요")
            if st.button("친구 불러오기", use_container_width=True) and load_friend(friend_input):
                st.rerun()

        # ----- 내 라이브러리 요약
        st.divider()
        owned = list(ss.profile["owned"].values())
        total_h = sum(g["playtime_forever"] for g in owned) / 60
        unplayed = sum(1 for g in owned if g["playtime_forever"] == 0)
        c1, c2 = st.columns(2)
        c1.metric("보유 게임", f"{len(owned)}개")
        c2.metric("총 플레이", f"{total_h:,.0f}시간")
        st.metric("사놓고 안 한 게임", f"{unplayed}개")

        st.subheader("많이 한 게임")
        top = sorted(owned, key=lambda g: g["playtime_forever"], reverse=True)[:10]
        df = pd.DataFrame({
            "게임": [g["name"] for g in top],
            "시간": [round(g["playtime_forever"] / 60, 1) for g in top],
        })
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={"시간": st.column_config.ProgressColumn(
                "플레이 시간", format="%.0fh", min_value=0, max_value=float(df["시간"].max() or 1))},
        )

        # ----- 찜 목록
        st.divider()
        st.subheader(f"⭐ 찜 목록 ({len(ss.wishlist)})")
        if not ss.wishlist:
            st.caption("추천 카드의 ⭐ 찜 버튼을 누르면 여기에 모여요.")
        for name, g in list(ss.wishlist.items()):
            c1, c2 = st.columns([5, 1])
            sale = f" · 🔥{g['discount']}%" if g.get("discount") else ""
            c1.markdown(f"[{name}]({g['store_url']}){sale}")
            c2.button("✕", key=f"unwish_{g['appid']}", on_click=toggle, args=("wishlist", g),
                      help="찜 해제")
        if ss.wishlist:
            csv = pd.DataFrame([{"게임": n, "현재가": g.get("price_now") or "", "상점": g["store_url"]}
                                for n, g in ss.wishlist.items()]).to_csv(index=False).encode("utf-8-sig")
            st.download_button("찜 목록 저장 (CSV)", csv, "찜목록.csv", "text/csv",
                               use_container_width=True)
            st.caption("찜 목록은 이 브라우저 탭에서만 유지돼요. 새로고침 전에 저장해 두세요.")

        # ----- 피드백 요약
        if ss.likes or ss.dislikes:
            st.divider()
            st.caption(f"👍 좋아요 {len(ss.likes)}개 · 👎 별로예요 {len(ss.dislikes)}개 → 다음 추천에 반영돼요")
            if st.button("평가 초기화", use_container_width=True):
                ss.likes, ss.dislikes = {}, {}
                st.rerun()

        st.divider()
        if st.button("대화 지우기", use_container_width=True):
            ss.messages, ss.memory = [], {}
            st.rerun()

# ---------------------------------------------------------------- 메인
st.title("이제뭐함")
mode = "  ·  👥 친구랑 같이 모드" if ss.friend else ""
st.caption("내 스팀 플레이 기록과 게임 정보를 바탕으로, 지금 할 게임을 골라주는 AI  ·  "
           f"남은 질문 {max(0, MAX_QUESTIONS - ss.asked)}개{mode}")


def price_line(g: dict) -> str:
    if g.get("free"):
        return "💸 **무료**"
    if not g.get("price_now"):
        return ""
    if g.get("discount"):
        return f"💰 ~~{g['price_orig']}~~ **{g['price_now']}** 🔥 **-{g['discount']}%**"
    return f"💰 {g['price_now']}"


def show_picks(picks: list, prefix: str):
    """추천 게임 카드: 이미지, 가격·할인, 상점/영상 링크, 좋아요·별로예요·찜 버튼"""
    if not picks:
        return
    st.markdown("##### 🎮 추천 게임 바로가기")
    cols = st.columns(2)
    for i, g in enumerate(picks):
        with cols[i % 2].container(border=True):
            st.image(g["image"])
            st.markdown(f"**{i + 1}위 · {g['name']}**")
            line = price_line(g)
            if line:
                st.markdown(line)
            info = [f"👍 긍정 {g['positive']:.0%}"]
            if g.get("median_hours"):
                info.append(f"⏱️ 보통 {g['median_hours']:.0f}시간")
            info.append("🎁 내가 보유" if g.get("is_owned") else "🛒 미보유")
            if g.get("friend_owned"):
                info.append("👥 친구도 보유" if g.get("is_owned") else "👥 친구가 보유")
            st.caption(" · ".join(info))
            st.caption(g["genres"])

            b1, b2 = st.columns(2)
            b1.link_button("스팀 상점", g["store_url"], use_container_width=True)
            b2.link_button("🎬 영상 보기",
                           "https://www.youtube.com/results?search_query="
                           + quote_plus(f"{g['name']} trailer"),
                           use_container_width=True)

            key = f"{prefix}_{g['appid']}"
            liked, disliked, wished = g["name"] in ss.likes, g["name"] in ss.dislikes, g["name"] in ss.wishlist
            f1, f2, f3 = st.columns(3)
            f1.button("👍 좋아요" + (" ✓" if liked else ""), key=f"{key}_like",
                      type="primary" if liked else "secondary", use_container_width=True,
                      on_click=toggle, args=("likes", g))
            f2.button("👎 별로" + (" ✓" if disliked else ""), key=f"{key}_dislike",
                      type="primary" if disliked else "secondary", use_container_width=True,
                      on_click=toggle, args=("dislikes", g))
            f3.button("⭐ 찜" + (" ✓" if wished else ""), key=f"{key}_wish",
                      type="primary" if wished else "secondary", use_container_width=True,
                      on_click=toggle, args=("wishlist", g))


def show_details(d: dict, prefix: str):
    show_picks(d.get("picks", []), prefix)
    with st.expander("🔎 검색 과정 보기"):
        st.markdown("**검색어:** " + " / ".join(f"`{q}`" for q in d.get("queries", [])))
        if d.get("relax_count"):
            st.markdown(f"**조건 완화:** {d['relax_count']}번 (후보가 부족해서 조건을 넓혀 다시 검색)")
        st.markdown("**후보 게임:**")
        st.dataframe(pd.DataFrame(d.get("candidates", [])), hide_index=True, use_container_width=True)


for idx, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("details"):
            show_details(m["details"], prefix=f"m{idx}")

if not ss.profile:
    st.info("👈 왼쪽에서 Steam ID를 입력하고 **불러오기**를 눌러주세요.")

limit_reached = ss.asked >= MAX_QUESTIONS
if ss.profile and limit_reached:
    st.warning(f"질문은 한 번에 {MAX_QUESTIONS}개까지 할 수 있어요. 잠시 후 페이지를 새로고침해 주세요.")
placeholder = "예: 둘이 같이 할 생존 게임 추천해줘" if ss.friend else "예: 친구랑 같이 할 협동 게임 추천해줘"
question = st.chat_input(placeholder, disabled=ss.profile is None or limit_reached)
if not question and ss.get("pending"):
    question = ss.pop("pending")  # 예시 버튼으로 고른 질문

if ss.profile and not ss.messages and not question:
    st.markdown("##### 이렇게 물어보세요")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            ss.pending = ex
            st.rerun()  # 버튼을 지우고 질문으로 처리

if question and not limit_reached:
    ss.asked += 1
    msg_idx = len(ss.messages) + 1  # 답변 메시지가 들어갈 위치 (버튼 이름에 사용)
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("질문을 분석하는 중...", expanded=False)
        box = st.empty()
        buf = []

        def on_token(text: str):
            buf.append(text)
            box.markdown("".join(buf) + "▌")

        agent.on_token = on_token
        final = {"question": question, "steamid": ss.steamid, "profile": ss.profile,
                 "history": ss.memory.get("history", []),
                 "exclude": ss.memory.get("exclude", []) + list(ss.wishlist),
                 "count": ss.get("count", agent.DEFAULT_COUNT),
                 "likes": list(ss.likes.values()), "dislikes": list(ss.dislikes.values())}
        if ss.friend:
            final["friend"] = ss.friend
        answer, details = None, None
        try:
            for chunk in agent.app.stream(dict(final), stream_mode="updates"):
                for node, upd in chunk.items():
                    final.update(upd or {})
                    status.write(f"✅ {agent.LABELS.get(node, node)}")
                    if node == "analyze":
                        status.update(label="비슷한 게임과 현재 가격을 찾는 중...")
                    elif node == "relax":
                        status.update(label="후보가 부족해서 조건을 넓혀 다시 검색하는 중...")
                    elif node == "retrieve":
                        status.update(label="추천 답변을 작성하는 중...")
            status.update(label="추천 완료", state="complete")
            answer = final.get("answer", "")
            agent.remember(ss.memory, question, final)
            if final.get("plan"):
                details = {
                    "picks": agent.ranked_picks(answer, final.get("candidates", [])),
                    "queries": final["plan"].get("queries", []),
                    "relax_count": final.get("relax_count", 0),
                    "candidates": [
                        {"게임": c["name"], "보유": c["owned"], "현재가": agent.price_text(c),
                         "긍정 평가": f"{c['positive']:.0%}", "장르": c["genres"]}
                        for c in final.get("candidates", [])
                    ],
                }
        except agent.errors.APIError as e:
            status.update(label="오류", state="error")
            answer = (f"Gemini가 지금 응답하지 못했어요 (오류 {e.code}). "
                      "사용자가 몰리는 중일 수 있으니 잠시 후 다시 물어봐 주세요.")
        except Exception as e:
            status.update(label="오류", state="error")
            answer = f"문제가 생겼어요: {type(e).__name__}: {e}"
        finally:
            agent.on_token = None

        box.markdown(answer)
        if details:
            show_details(details, prefix=f"m{msg_idx}")

    ss.messages.append({"role": "assistant", "content": answer, "details": details})
