import streamlit as st

from conductor import Conductor
from prompts import scenarios, get_scenario

st.title("AI Lawyers Battle Royale")


def reset():
    del st.session_state["messages"]


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="key_openai", type="password")

    selected_scenario = st.selectbox('Select scenario', [scenario.rule_title for scenario in scenarios],
                                     on_change=reset)

    st.button('Reset', on_click=reset)

    autopilot = st.checkbox('Autopilot', False, help="Runs AI Models against each other as Plaintiff and Defendant")

    if selected_scenario:
        st.header('Selected scenario')
        scenario = get_scenario(selected_scenario).dict()
        for key in scenario.keys():
            st.write(f"**{key}**:", scenario[key])

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

conductor = Conductor(openai_api_key, get_scenario(selected_scenario), autopilot)

for msg in st.session_state.messages:
    if msg["role"] == 'court':
        st.chat_message(msg["role"], avatar="🧑‍⚖️").write(msg["content"])
    elif msg["role"] == 'counsel':
        st.chat_message(msg["role"], avatar="🤖️️️").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="👦").write(msg["content"])

conductor.run()

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👦").write(prompt)

    conductor.run()
