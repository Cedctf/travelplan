from datetime import date
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from src.orchestration.graph import build_graph
from src.orchestration.state import new_state

ICONS = {"planner": "🧠", "flight": "🛫", "hotel": "🏨", "itinerary": "🗺️"}

st.set_page_config(page_title="Agentic Travel Planner", layout="wide")


def init_state():
    st.session_state.setdefault("stage", "idle")
    st.session_state.setdefault("trace", [])
    st.session_state.setdefault("app", None)
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("snapshot", {})
    st.session_state.setdefault("interrupt", None)


def render_entry(entry, container):
    icon = ICONS.get(entry["agent"], "•")
    with container:
        st.markdown(f"**{icon} {entry['agent']}** — {entry['thought'] or ''}")
        if entry["action"]:
            st.code(entry["action"], language="text")
        if entry["observation"]:
            obs = entry["observation"]
            st.caption("↳ " + (obs[:280] + "…" if len(obs) > 280 else obs))


def drive(stream, status):
    entries = st.session_state.trace
    for chunk in stream:
        if "__interrupt__" in chunk:
            st.session_state.interrupt = chunk["__interrupt__"][0].value
            continue
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for entry in update.get("reasoning_trace", []):
                entries.append(entry)
                render_entry(entry, status)
    st.session_state.trace = entries


def run_planning(request, traveller):
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
    st.session_state.app = app
    st.session_state.config = config
    st.session_state.trace = []
    st.session_state.interrupt = None
    with st.status("Agents are planning (live sandbox)…", expanded=True) as status:
        drive(app.stream(new_state(request, traveller), config, stream_mode="updates"), status)
        status.update(label="Planning complete", state="complete")
    st.session_state.snapshot = app.get_state(config).values
    st.session_state.stage = "planned" if app.get_state(config).next else "ended"


def run_resume(decision):
    app = st.session_state.app
    config = st.session_state.config
    with st.status(f"Resuming with '{decision}'…", expanded=True) as status:
        drive(app.stream(Command(resume=decision), config, stream_mode="updates"), status)
        status.update(label="Done", state="complete")
    st.session_state.snapshot = app.get_state(config).values
    st.session_state.stage = "booked"


def render_constraints(s):
    st.subheader("Parsed request")
    cols = st.columns(4)
    cols[0].metric("From", s.get("origin") or "—")
    cols[1].metric("To", s.get("destination") or "—")
    dates = s.get("dates") or {}
    cols[2].metric("Nights", dates.get("nights") or "—")
    cols[3].metric("Travellers", s.get("travellers") or "—")
    if s.get("constraints"):
        st.write("**Constraints:** " + ", ".join(s["constraints"]))
    if s.get("preferences"):
        st.write("**Preferences:** " + ", ".join(s["preferences"]))


def render_budget(s):
    st.subheader("Budget")
    cols = st.columns(3)
    cols[0].metric("Total budget", s.get("budget_total"))
    cols[1].metric("Estimated cost", s.get("estimated_total_cost"))
    remaining = s.get("budget_remaining")
    cols[2].metric("Remaining", remaining,
                   delta=None if remaining is None else round(remaining, 2))
    alloc = s.get("budget_allocations") or {}
    if alloc:
        st.write("**Allocations:** "
                 + " · ".join(f"{k}: {v}" for k, v in alloc.items()))
    history = s.get("replanning_history") or []
    if history:
        st.write("**Replanning timeline:**")
        for i, h in enumerate(history, 1):
            st.write(f"{i}. est {round(h['estimated_total_cost'], 2)} · "
                     f"over by {round(h['over_by'], 2)} → re-dispatch **{h['target']}**")


def render_summary(s):
    flight = s.get("selected_flight") or {}
    hotel = s.get("selected_hotel") or {}
    cols = st.columns(2)
    cols[0].markdown(f"🛫 **Flight** — {flight.get('airline', '—')} · "
                     f"{flight.get('price', '—')} {flight.get('currency', '')}")
    cols[1].markdown(f"🏨 **Hotel** — {hotel.get('name', '—')} · "
                     f"{hotel.get('price', '—')} {hotel.get('currency', '')}")
    itinerary = (s.get("selected_itinerary") or {}).get("plan")
    if itinerary:
        with st.expander("🗺️ Itinerary"):
            st.markdown(itinerary)


def render_booking(s):
    status = s.get("booking_status") or {}
    overall = status.get("overall")
    if overall == "confirmed":
        st.success("Booking confirmed ✅")
    elif overall == "blocked":
        st.warning("Booking blocked — approval was not granted.")
    else:
        st.error("Booking did not fully complete.")
    st.json(status)


init_state()
st.title("✈️ Agentic Travel Planner")
st.caption("Planner + Flight / Hotel / Itinerary specialists · live sandbox APIs")

request = st.text_area(
    "Travel request",
    value="Plan a 5-day trip to Tokyo from Singapore in December for 2 people "
          "with a budget of USD 4000. We love anime, food and nature. Nonstop flights.",
    height=90,
)

with st.expander("👤 Traveller details (used when booking)", expanded=True):
    c1, c2 = st.columns(2)
    given_name = c1.text_input("First name", value="")
    family_name = c2.text_input("Last name", value="")
    c3, c4 = st.columns(2)
    email = c3.text_input("Email", value="")
    phone_number = c4.text_input("Phone number", value="", placeholder="+65...")
    c5, c6, c7 = st.columns(3)
    title = c5.selectbox("Title", ["mr", "mrs", "ms", "miss", "dr"])
    gender = c6.selectbox("Gender", ["m", "f"])
    born_on = c7.date_input(
        "Date of birth",
        value=date(1990, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today(),
    )


def collect_traveller():
    return {
        "title": title,
        "given_name": given_name,
        "family_name": family_name,
        "gender": gender,
        "born_on": born_on.isoformat(),
        "email": email,
        "phone_number": phone_number,
        # hotel provider uses camelCase names
        "firstName": given_name,
        "lastName": family_name,
    }


def missing_details(traveller):
    required = ["given_name", "family_name", "email", "phone_number"]
    return [f for f in required if not traveller.get(f)]


if st.button("Plan trip", type="primary", disabled=st.session_state.stage == "running"):
    traveller = collect_traveller()
    missing = missing_details(traveller)
    if missing:
        st.error("Please fill in your traveller details before planning: "
                 + ", ".join(f.replace("_", " ") for f in missing))
    else:
        run_planning(request, traveller)
        st.rerun()

snap = st.session_state.snapshot
if snap:
    render_constraints(snap)
    render_budget(snap)
    st.subheader("Agent reasoning")
    box = st.container()
    for entry in st.session_state.trace:
        render_entry(entry, box)

if st.session_state.stage == "planned":
    st.divider()
    st.subheader("Approval required")
    render_summary(snap)
    c1, c2 = st.columns(2)
    if c1.button("✅ Book this itinerary", type="primary"):
        run_resume("approved")
        st.rerun()
    if c2.button("❌ Cancel"):
        run_resume("rejected")
        st.rerun()

if st.session_state.stage == "booked":
    st.divider()
    render_booking(snap)

if st.session_state.stage == "ended":
    st.divider()
    st.warning("No feasible plan within budget.")
    for note in snap.get("planner_notes", []):
        st.write("• " + note)
