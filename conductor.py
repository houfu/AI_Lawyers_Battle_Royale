from typing import Dict, Any, List, Optional
from typing import Literal
from uuid import UUID

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from prompts import Scenario, plaintiff_template, court_template, costs_system_template, costs_user_template, \
    defendant_template


class StreamHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        pass

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.write(self.text)


def add_message(role: Literal['court', 'counsel', 'user']):
    if role == 'court':
        with st.chat_message("court", avatar="🧑‍⚖️"):
            return st.empty()
    elif role == 'counsel':
        with st.chat_message("counsel", avatar="🤖️"):
            return st.empty()
    else:
        with st.chat_message("user", avatar="👦️"):
            return st.empty()


def costs_determination(messages):
    system_message_prompt = SystemMessagePromptTemplate.from_template(costs_system_template)
    user_message_prompt = HumanMessagePromptTemplate.from_template(costs_user_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    decision = messages[-1]["content"]
    transcript = ""
    for argument in messages[:-1]:
        transcript += argument["content"]
    return chat_prompt.format_prompt(
        decision=decision, transcript=transcript
    ).to_messages()


class Conductor:
    def __init__(self, openai_key: str, scenario: Scenario, autopilot=False):
        self.openai_key = openai_key
        if "messages" not in st.session_state:
            initialize()
        self.scenario = scenario.dict()
        self.autopilot = autopilot

    def run(self):
        last_message = st.session_state.messages[-1]

        if last_message["role"] == "court":
            if last_message["content"][-4:] == '[PC]':
                stream_handler = StreamHandler(add_message('counsel'))
                llm = ChatOpenAI(
                    temperature=0.6,
                    openai_api_key=self.openai_key, streaming=True, callbacks=[stream_handler]
                )
                response = llm(self.convert_messages_to_langchain_schema(st.session_state.messages, 'counsel'))
                st.session_state.messages.append({"role": "counsel", "content": response.content})
                self.run()
            if last_message["content"][-4:] == '[DC]' and self.autopilot:
                stream_handler = StreamHandler(add_message('user'))
                llm = ChatOpenAI(
                    temperature=0.6,
                    openai_api_key=self.openai_key, streaming=True, callbacks=[stream_handler]
                )
                response = llm(self.convert_messages_to_langchain_schema(st.session_state.messages, 'user'))
                st.session_state.messages.append({"role": "user", "content": response.content})
                self.run()
            if last_message["content"][-5:] == '[END]':
                stream_handler = StreamHandler(add_message('court'))
                llm = ChatOpenAI(
                    model_name="gpt-4", temperature=0.2,
                    openai_api_key=self.openai_key, streaming=True, callbacks=[stream_handler]
                )
                response = llm(costs_determination(st.session_state.messages))
                st.session_state.messages.append({"role": "court", "content": response.content})
        else:
            stream_handler = StreamHandler(add_message('court'))
            llm = ChatOpenAI(
                temperature=0.4,
                openai_api_key=self.openai_key, streaming=True, callbacks=[stream_handler]
            )
            response = llm(self.convert_messages_to_langchain_schema(st.session_state.messages, 'court'))
            st.session_state.messages.append({"role": "court", "content": response.content})
            self.run()

    def convert_messages_to_langchain_schema(self, messages, role: str):
        if role == 'counsel':
            template = plaintiff_template
        elif role == 'court':
            template = court_template
        else:
            template = defendant_template
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        scenario = self.scenario
        result = [system_message_prompt.format(**scenario)]
        for message in messages:
            message_role = message["role"]
            if message_role == role:
                result.append(AIMessage(content=message["content"]))
            elif message_role == 'court':
                content = f"CT: {message['content']}"
                result.append(HumanMessage(content=content))
            elif message_role == 'counsel':
                content = f"PC: {message['content']}"
                result.append(HumanMessage(content=content))
            else:
                content = f"DC: {message['content']}"
                result.append(HumanMessage(content=content))
        return result


def initialize():
    st.session_state["messages"] = [
        {"role": "court", "content": "Plaintiff's counsel, you may now begin. [PC]"}
    ]
