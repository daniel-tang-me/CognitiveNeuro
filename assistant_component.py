"""Contextual CognitiveNeuro assistant available throughout the application."""

import streamlit as st

from ai_service import answer_question_stream


SUGGESTED_PROMPTS = (
    "Explain this result",
    "Interpret EEG bands",
    "What should I discuss with a clinician?",
)


@st.fragment
def render_analysis_assistant():
    """Render a persistent floating popover backed by Streamlit session state."""
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] {
            position:fixed !important; inset:auto 24px 48px auto !important;
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
        .sage-question {
            display:inline-block; background:#dceff7; color:#000;
            border-radius:999px; padding:.34rem .7rem; margin-bottom:.8rem;
            font-size:.72rem; font-weight:600;
        }
        .sage-greeting {
            color:#000; font-size:1.5rem; line-height:1.2; margin:0 0 .24rem;
        }
        .sage-help {
            color:#000; font-size:.88rem; line-height:1.4; margin:0 0 .85rem;
        }
        div[data-testid="stPopoverBody"] h4 { font-size:1rem !important; margin-bottom:.2rem; }
        div[data-testid="stPopoverBody"] [data-testid="stButton"] button {
            width:100%; justify-content:flex-start; background:transparent; border:0;
            border-bottom:1px solid #dce5e7; border-radius:0 !important; padding:.5rem 0;
            color:#486a7e; font-size:.8rem; min-height:auto;
        }
        div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"] {
            justify-content:center; background:#dceff7 !important; color:#000 !important;
            border:1px solid #bfdde9 !important; border-radius:6px !important;
            padding:.55rem .75rem; margin-bottom:.45rem;
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
                inset:auto 16px 42px auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    assistant_popover = st.popover(
        " ",
        icon=":material/neurology:",
        key="analysis_assistant_popover",
        on_change="rerun",
    )
    with assistant_popover:
        _enable_chat_autoscroll(scroll_on_open=bool(assistant_popover.open))
        st.markdown(
            '<div class="sage-question">Any questions?</div>'
            '<div class="sage-greeting">Hi, I\'m <strong>Sage</strong>.</div>'
            '<div class="sage-help">What can I help you with?</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Go to current EEG analysis",
            key="assistant_current_analysis",
            width="stretch",
            type="primary",
        ):
            st.session_state.page = "EEG Analysis"
            st.session_state.top_navigation = "EEG Analysis"
            st.rerun(scope="app")

        context = st.session_state.get("analysis_context")
        if context:
            context = {**context, "page": st.session_state.get("page", "EEG Analysis")}
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
                    _submit_streaming(prompt, context)
                    return

        for message in messages[-6:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Ask about this analysis…", key="analysis_assistant_input")
        if prompt:
            _submit_streaming(prompt, context)
            return


def _submit_streaming(prompt: str, context: dict | None):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        stream_marker = st.empty()
        stream_marker.markdown(
            '<span id="sage-stream-active" style="display:none"></span>',
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Sage is reviewing the current analysis…"):
                response = st.write_stream(answer_question_stream(prompt, context, history=history))
        except Exception:
            response = (
                "The CognitiveNeuro Assistant is temporarily unavailable. Your analysis "
                "context remains saved; please try again shortly."
            )
            st.write(response)
        finally:
            stream_marker.empty()
    st.session_state.messages.append({"role": "assistant", "content": response})


def _enable_chat_autoscroll(scroll_on_open: bool = False):
    """Keep the open assistant popover pinned to new streamed content."""
    st.html(
        """
        <script>
        (() => {
          const parentDocument = window.parent.document;
          const scrollOnOpen = __SCROLL_ON_OPEN__;
          const findPopover = () => parentDocument.querySelector(
            'div[data-testid="stPopoverBody"]'
          );
          let observed = null;
          let observer = null;

          const scrollToBottom = (popover) => {
            let scroller = popover;
            let candidate = popover;
            while (candidate && candidate !== parentDocument.body) {
              if (candidate.scrollHeight > candidate.clientHeight) {
                const overflow = parentDocument.defaultView
                  .getComputedStyle(candidate).overflowY;
                if (overflow === 'auto' || overflow === 'scroll') {
                  scroller = candidate;
                  break;
                }
              }
              candidate = candidate.parentElement;
            }
            scroller.scrollTop = scroller.scrollHeight;
          };

          const attach = () => {
            const popover = findPopover();
            if (!popover || popover === observed) return;
            if (observer) observer.disconnect();
            observed = popover;
            if (scrollOnOpen) {
              requestAnimationFrame(() => scrollToBottom(popover));
            }
            observer = new MutationObserver(() => {
              requestAnimationFrame(() => {
                if (!popover.querySelector('#sage-stream-active')) return;
                scrollToBottom(popover);
              });
            });
            observer.observe(popover, {
              childList: true,
              subtree: true,
              characterData: true
            });
          };

          attach();
          const finder = window.setInterval(attach, 250);
          window.setTimeout(() => window.clearInterval(finder), 120000);
        })();
        </script>
        """.replace("__SCROLL_ON_OPEN__", "true" if scroll_on_open else "false"),
        unsafe_allow_javascript=True,
    )
