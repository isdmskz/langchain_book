import os
from dotenv import load_dotenv
load_dotenv() # .envに、OpenAI/GoogleのAPIkey, ID 諸々記載

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


def create_agent_chain(): # ChatモデルとMemoryを定義してAgent初期化
    
    # OpenAI Chat Completions API の仕様設定（.envからmodel, temperatureを読込）
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True, # ストリーミング表示
    )
    
    # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    
    # OpenAI Functions Agentが使える設定でMemoryを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    tools = load_tools(["google-search", "wikipedia"])
    
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS, # 安定動作のOpenAI Functions Agent(Function callingに対応したAgent)
        agent_kwargs=agent_kwargs, # いったんデフォルト引数を宣言すると、その後は同語であってもデフォルト宣言が必要
        memory=memory, # 会話履歴を踏まえて応答させるための部品
    )


# 会話履歴を踏まえて応答させるための部品
if "agent_chain" not in st.session_state: # st.session_stateに"agent_chain"がない場合
    st.session_state.agent_chain = create_agent_chain() # st.session_stateを使って一度だけAgentを初期化



st.title("LangChainチャットボット")



# 初期設定
if "messages" not in st.session_state: # st.session_stateに"messages"がない場合
    st.session_state.messages = [] # st.session_state.messagesを空のリストで初期化
    
for message in st.session_state.messages: # st.session_state.messagesでループ
    with st.chat_message(message["role"]): # 役割毎に
        st.markdown(message["content"]) # 保存されているmessage内容をmarkdownとして整形して表示

# userからの入力を受け付ける
prompt = st.chat_input("入力してください")

if prompt: # promptに入力された文字列がある（Noneでも空文字列でもない）場合

    # 会話履歴を残すための部品（userのpromptをst.session_state.messagesに追加）
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"): # userのアイコン内に
        st.markdown(prompt) # promptをmarkdownとして整形して表示
    
    with st.chat_message("assistant"): # AIアシスタントのアイコン内に
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback]) # 会話履歴を踏まえて応答させるためst.session_state.agent_chainを指定
        st.markdown(response) # responseをmarkdownとして整形して表示
    
    # 会話履歴を残すための部品（AIエージェントの応答内容をst.session_state.messagesに追加）
    st.session_state.messages.append({"role": "assistant", "content": response})

