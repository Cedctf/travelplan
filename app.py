from dotenv import load_dotenv
load_dotenv()

import re
from datetime import date
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from src.orchestration.graph import build_graph
from src.orchestration.state import new_state

st.set_page_config(page_title="Agentic Travel Planner", layout="wide")


def init_state():
    st.session_state.setdefault("stage", "idle")
    st.session_state.setdefault("trace", [])
    st.session_state.setdefault("app", None)
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("snapshot", {})
    st.session_state.setdefault("interrupt", None)


def render_entry(entry, container):
    with container:
        st.markdown(f"**{entry['agent']}** — {entry['thought'] or ''}")
        if entry["action"]:
            st.code(entry["action"], language="text")
        if entry["observation"]:
            obs = entry["observation"]
            st.caption("↳ " + (obs[:280] + "…" if len(obs) > 280 else obs))


def drive(stream, status):
    entries = st.session_state.trace
    for mode, chunk in stream:
        if mode == "custom":
            entries.append(chunk)
            render_entry(chunk, status)
        elif mode == "updates" and "__interrupt__" in chunk:
            st.session_state.interrupt = chunk["__interrupt__"][0].value
    st.session_state.trace = entries


def run_planning(request, container):
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 60}
    st.session_state.app = app
    st.session_state.config = config
    st.session_state.trace = []
    st.session_state.interrupt = None
    with container, st.status("Agents are planning (live sandbox)…",
                              expanded=True) as status:
        drive(app.stream(new_state(request), config,
                         stream_mode=["updates", "custom"]), status)
        status.update(label="Planning complete", state="complete")
    st.session_state.snapshot = app.get_state(config).values
    st.session_state.stage = "planned" if app.get_state(config).next else "ended"


def run_resume(decision, container, traveller=None):
    app = st.session_state.app
    config = st.session_state.config
    if traveller:
        app.update_state(config, {"traveller": traveller})
    with container, st.status(f"Resuming with '{decision}'…",
                              expanded=True) as status:
        drive(app.stream(Command(resume=decision), config,
                         stream_mode=["updates", "custom"]), status)
        status.update(label="Done", state="complete")
    st.session_state.snapshot = app.get_state(config).values
    st.session_state.stage = "booked"


def render_constraints(s):
    st.subheader("Parsed request")
    dates = s.get("dates") or {}
    st.markdown(
        f"**From:** {s.get('origin') or '—'} · "
        f"**To:** {s.get('destination') or '—'} · "
        f"**Nights:** {dates.get('nights') or '—'} · "
        f"**Travellers:** {s.get('travellers') or '—'}"
    )
    if s.get("constraints"):
        st.write("**Constraints:** " + ", ".join(s["constraints"]))
    if s.get("preferences"):
        st.write("**Preferences:** " + ", ".join(s["preferences"]))


def render_budget(s):
    st.subheader("Budget")
    remaining = s.get("budget_remaining")
    st.markdown(
        f"**Total budget:** {s.get('budget_total')} · "
        f"**Estimated cost:** {s.get('estimated_total_cost')} · "
        f"**Remaining:** {None if remaining is None else round(remaining, 2)}"
    )
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
    cols[0].markdown(f"**Flight** — {flight.get('airline', '—')} · "
                     f"{flight.get('price', '—')} {flight.get('currency', '')}")
    cols[1].markdown(f"**Hotel** — {hotel.get('name', '—')} · "
                     f"{hotel.get('price', '—')} {hotel.get('currency', '')}")
    itinerary = (s.get("selected_itinerary") or {}).get("plan")
    if itinerary:
        with st.expander("Itinerary"):
            st.markdown(
                f"**Estimated trip cost:** {s.get('estimated_total_cost', '—')} "
                f"({flight.get('price', '—')} {flight.get('currency', '')} · "
                f"{hotel.get('price', '—')} {hotel.get('currency', '')})"
            )
            st.markdown(itinerary)


def render_booking(s):
    status = s.get("booking_status") or {}
    overall = status.get("overall")
    if overall == "confirmed":
        st.success("Booking confirmed")
    elif overall == "blocked":
        st.warning("Booking blocked — approval was not granted.")
    else:
        st.error("Booking did not fully complete.")
    st.json(status)


init_state()
st.title("Agentic Travel Planner")
st.caption("Planner + Flight / Hotel / Itinerary specialists · live sandbox APIs")

live = st.container()

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def _adult_cutoff():
    """Latest DOB that still counts as an adult (18+), for the airline."""
    today = date.today()
    try:
        return today.replace(year=today.year - 18)
    except ValueError:
        return today.replace(year=today.year - 18, day=28)


def render_traveller_form():
    c1, c2 = st.columns(2)
    given_name = c1.text_input("First name", value="")
    family_name = c2.text_input("Last name", value="")
    c3, c4 = st.columns(2)
    email = c3.text_input("Email", value="")
    phone_number = c4.text_input("Phone number", value="+60",
                                 placeholder="+60123456789",
                                 help="International format with country code "
                                      "(Malaysia +60 by default), e.g. +60123456789")
    c5, c6, c7 = st.columns(3)
    title = c5.selectbox("Title", ["mr", "mrs", "ms", "miss", "dr"])
    gender = c6.selectbox("Gender", ["m", "f"])
    born_on = c7.date_input(
        "Date of birth",
        value=date(1990, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=_adult_cutoff(),
    )
    phone_number = phone_number.replace(" ", "").replace("-", "")
    return {
        "title": title,
        "given_name": given_name,
        "family_name": family_name,
        "gender": gender,
        "born_on": born_on.isoformat(),
        "email": email,
        "phone_number": phone_number,
        "firstName": given_name,
        "lastName": family_name,
    }


def traveller_errors(traveller):
    errors = []
    required = ["given_name", "family_name", "email", "phone_number"]
    missing = [f for f in required if not traveller.get(f)]
    if missing:
        errors.append("Fill in: " + ", ".join(f.replace("_", " ") for f in missing))
    phone = traveller.get("phone_number") or ""
    if phone and not _E164_RE.match(phone):
        errors.append("Phone must be in international format with country code, "
                      "e.g. +6591234567")
    return errors


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
    st.markdown("**Traveller details** (used to complete the booking)")
    traveller = render_traveller_form()
    c1, c2 = st.columns(2)
    if c1.button("Book this itinerary", type="primary"):
        errors = traveller_errors(traveller)
        if errors:
            for err in errors:
                st.error(err)
        else:
            run_resume("approved", live, traveller)
            st.rerun()
    if c2.button("Cancel"):
        run_resume("rejected", live)
        st.rerun()

if st.session_state.stage == "booked":
    st.divider()
    render_booking(snap)

if st.session_state.stage == "ended":
    st.divider()
    st.warning("No feasible plan within budget.")
    for note in snap.get("planner_notes", []):
        st.write("• " + note)

st.markdown(
    """
    <style>
    .st-key-input_bar {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 100;
        margin: 0 auto;
        max-width: 46rem;
        padding: 0.75rem 1.25rem 1rem;
        background: var(--background-color, #fff);
    }
    .stMainBlockContainer { padding-bottom: 16rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="input_bar"):
    request = st.text_area(
        "Travel request",
        value="Plan a 3-day trip to Bangkok from Singapore in March for 1 person "
              "with a budget of USD 6000. We love street food and temples.",
        height=90,
    )
    if st.button("Plan trip", type="primary",
                 disabled=st.session_state.stage == "running",
                 use_container_width=True):
        run_planning(request, live)
        st.rerun()
