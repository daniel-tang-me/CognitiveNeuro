"""Contextual CognitiveNeuro assistant available throughout the application."""

import streamlit as st

from ai_service import answer_question


SUGGESTED_PROMPTS = (
    "Explain this result",
    "Interpret EEG bands",
    "What should I discuss with a clinician?",
)


def render_analysis_assistant():
    """Render a persistent floating popover backed by Streamlit session state."""
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] {
            position:fixed !important; inset:auto 24px 24px auto !important;
            width:48px !important; height:48px !important; margin:0 !important;
            z-index:9999 !important;
        }
        div[data-testid="stPopover"] button {
            width: 48px; height: 48px; min-height:48px; border-radius:50% !important;
            padding:0 !important; background:#ffffff !important; color:#efb59c !important;
            border:2px solid #efb59c !important; outline:1px solid rgba(255,255,255,.72);
            outline-offset:-4px; box-shadow:0 7px 20px rgba(39,67,84,.15);
            font-size:1.25rem; letter-spacing:-.15em; position:relative; overflow:visible;
        }
        div[data-testid="stPopover"] button:hover {
            background:#ffffff !important; transform:translateY(-1px);
            box-shadow:0 9px 22px rgba(39,67,84,.19);
        }
        div[data-testid="stPopover"] button p { display:none !important; }
        div[data-testid="stPopover"] button span[data-testid="stIconMaterial"],
        div[data-testid="stPopover"] button svg {
            display:none !important;
        }
        div[data-testid="stPopover"] button::before {
            content:"neurology"; color:#efb59c; font-family:"Material Symbols Rounded";
            font-size:23px; font-weight:400; font-style:normal; line-height:1;
            letter-spacing:normal; text-transform:none; white-space:nowrap;
            word-wrap:normal; direction:ltr; font-feature-settings:"liga";
            -webkit-font-feature-settings:"liga"; -webkit-font-smoothing:antialiased;
            position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
        }
        div[data-testid="stPopoverBody"] {
            width:min(390px,calc(100vw - 32px)); background:#f8f8f4;
            border:1px solid #cddadd; border-radius:4px; padding:1.15rem;
            box-shadow:0 16px 42px rgba(37,59,71,.16);
        }
        div[data-testid="stPopoverBody"] h4 { font-size:1rem !important; margin-bottom:.2rem; }
        div[data-testid="stPopoverBody"] [data-testid="stButton"] button {
            width:100%; justify-content:flex-start; background:transparent; border:0;
            border-bottom:1px solid #dce5e7; border-radius:0 !important; padding:.5rem 0;
            color:#486a7e; font-size:.8rem; min-height:auto;
        }
        div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] {
            background:transparent; border:0; border-top:1px solid #e0e7e9;
            border-radius:0; padding:.75rem 0; gap:.55rem;
        }
        div[data-testid="stPopoverBody"] [data-testid="stChatMessageAvatar"] {
            width:24px; height:24px; background:#e5eef1; border-radius:2px;
        }
        div[data-testid="stPopoverBody"] [data-testid="stChatInput"] {
            border-radius:2px; border-color:#cbdade; background:#fff;
        }
        @media (max-width:640px) {
            div[data-testid="stPopover"] {
                inset:auto 16px 18px auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.popover(
        " ",
        icon=":material/neurology:",
    ):
        st.markdown('<div class="eyebrow">Current EEG Analysis</div>', unsafe_allow_html=True)
        st.markdown("#### CognitiveNeuro Assistant")
        context = st.session_state.get("analysis_context")
        if context:
            st.caption(
                f"Remembered analysis · {context['source']} · "
                f"{context['result']['pattern_probability'] * 100:.1f}% pattern similarity"
            )
            st.caption(f"Experimental verdict: {context.get('verdict', 'available in analysis')}")
        else:
            st.caption("Available on every page. Run an analysis to attach EEG context.")

        messages = st.session_state.setdefault("messages", [])
        if not messages:
            st.markdown("**Ask about this EEG analysis**")
            st.caption("The assistant explains experimental output and uncertainty; it does not diagnose or prescribe.")
            for index, prompt in enumerate(SUGGESTED_PROMPTS):
                if st.button(prompt, key=f"assistant_suggestion_{index}", width="stretch"):
                    _submit(prompt, context)
                    st.rerun()

        for message in messages[-6:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Ask about this analysis…", key="analysis_assistant_input")
        if prompt:
            _submit(prompt, context)
            st.rerun()


def _submit(prompt: str, context: dict | None):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = answer_question(prompt, context, history=history)
    except Exception:
        response = (
            "The CognitiveNeuro Assistant is temporarily unavailable. Your analysis "
            "context remains saved; please try again shortly."
        )
    st.session_state.messages.append({"role": "assistant", "content": response})
