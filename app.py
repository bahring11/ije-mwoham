"""이제뭐함 - Streamlit 웹 화면
실행: streamlit run app.py
"""
import os

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
    "내가 가진 게임 중에 아직 안 해본 거",
]

ss = st.session_state
ss.setdefault("profile", None)
ss.setdefault("steamid", None)
ss.setdefault("messages", [])
ss.setdefault("auto_tried", False)
ss.setdefault("asked", 0)
ss.setdefault("memory", {})  # 대화 기억 (이전 질문, 이미 추천한 게임)


def load(user_input: str):
    with st.spinner("플레이 기록 불러오는 중..."):
        sid = resolve_steamid(user_input) if user_input.strip() else None
        prof = agent.load_profile({"steamid": sid}).get("profile") if sid else None
    if prof:
        ss.profile, ss.steamid, ss.messages, ss.memory = prof, sid, [], {}
    else:
        st.error("플레이 기록을 불러오지 못했어요. Steam ID와 프로필 공개 설정을 확인하세요.")


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
        if st.button("대화 지우기", use_container_width=True):
            ss.messages, ss.memory = [], {}
            st.rerun()

# ---------------------------------------------------------------- 메인
st.title("이제뭐함")
st.caption("내 스팀 플레이 기록과 게임 정보를 바탕으로, 지금 할 게임을 골라주는 AI  ·  "
           f"남은 질문 {max(0, MAX_QUESTIONS - ss.asked)}개")


def show_details(d: dict):
    with st.expander("🔎 검색 과정 보기"):
        st.markdown("**검색어:** " + " / ".join(f"`{q}`" for q in d.get("queries", [])))
        if d.get("relax_count"):
            st.markdown(f"**조건 완화:** {d['relax_count']}번 (후보가 부족해서 조건을 넓혀 다시 검색)")
        st.markdown("**후보 게임:**")
        st.dataframe(pd.DataFrame(d.get("candidates", [])), hide_index=True, use_container_width=True)


for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("details"):
            show_details(m["details"])

if not ss.profile:
    st.info("👈 왼쪽에서 Steam ID를 입력하고 **불러오기**를 눌러주세요.")

limit_reached = ss.asked >= MAX_QUESTIONS
if ss.profile and limit_reached:
    st.warning(f"질문은 한 번에 {MAX_QUESTIONS}개까지 할 수 있어요. 잠시 후 페이지를 새로고침해 주세요.")
question = st.chat_input("예: 친구랑 같이 할 협동 게임 추천해줘",
                         disabled=ss.profile is None or limit_reached)
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
                 "history": ss.memory.get("history", []), "exclude": ss.memory.get("exclude", [])}
        answer, details = None, None
        try:
            for chunk in agent.app.stream(dict(final), stream_mode="updates"):
                for node, upd in chunk.items():
                    final.update(upd or {})
                    status.write(f"✅ {agent.LABELS.get(node, node)}")
                    if node == "analyze":
                        status.update(label="비슷한 게임을 검색하는 중...")
                    elif node == "relax":
                        status.update(label="후보가 부족해서 조건을 넓혀 다시 검색하는 중...")
                    elif node == "retrieve":
                        status.update(label="추천 답변을 작성하는 중...")
            status.update(label="추천 완료", state="complete")
            answer = final.get("answer", "")
            agent.remember(ss.memory, question, final)
            if final.get("plan"):
                details = {
                    "queries": final["plan"].get("queries", []),
                    "relax_count": final.get("relax_count", 0),
                    "candidates": [
                        {"게임": c["name"], "보유": c["owned"], "긍정 평가": f"{c['positive']:.0%}",
                         "장르": c["genres"]}
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
            show_details(details)

    ss.messages.append({"role": "assistant", "content": answer, "details": details})
