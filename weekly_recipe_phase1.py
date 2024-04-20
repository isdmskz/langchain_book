import os
from dotenv import load_dotenv
load_dotenv()
import webbrowser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
import streamlit as st



## ★メインエリア ##

# 横幅いっぱいに表示
st.set_page_config(layout="wide")

# タブ分け
tab1, tab2, tab3 = st.tabs(["履歴管理","今週の献立","買い物リスト"])

with tab1:
    st.header("前週の献立を履歴に移動して保存")
    
    col1, col2 = st.columns((1,1))
    with col1:
        st.text_area("履歴")

    with col2:
        st.text_area("前週食べたもの")
        st.button("履歴に移動して消去")
    
with tab2:
    st.header("向こう一週間の献立を考える")
    
    col1, col2 = st.columns((1,1))
    with col1:
        st.text_area("お気に入りレシピ")
        st.button("お気に入り 再読込")
    with col2:
        st.text_area("今週の献立を考える")
        st.button("今週の献立 更新")
        st.button("今週の献立 再読込")    
with tab3:
    st.header("買い物リストを作る")
    
    col1, col2 = st.columns((1,1))
    with col1:
        st.text_area("今週の献立")

    with col2:
        st.text_area("買い物リスト")
        st.button("買い物リスト 更新")








## ★サイドバーの料理検索チャットボット部分 ##


# {site}の定義
select_site = st.sidebar.selectbox("検索したいサイト:",("クックパッド", "マカロニ"))
if select_site == "クックパッド":
    site = "クックパッド 殿堂入り"
elif select_site == "マカロニ":
    site = "マカロニ Youtube"

# {material}の定義
material = st.sidebar.text_input("食材：")
retrieve = st.sidebar.button("検索")


def retrieve_recipe(site, material):

    # ChatPromptTemplateの定義
    system_template="""
    あなたは、与えられた条件をもとにGoogleでレシピ検索を実行して回答を返すことが得意な、忠実なアシスタントです。
    """

    human_template="""
    以下の「検索条件」をキーワードとしてGoogle検索を実行し、回答は「回答ルール」に従うこと。

    検索条件: {site} {material} レシピ

    回答ルール: Google検索結果のページのみ表示
    [Google Search Result]といったheaderを付けることを禁止する。
    Google検索結果ページのURLを""で囲った文字列のみ回答すること。
    URL以外の余計な情報は付与してはいけない。
    Google検索結果からHumanが選ぶため、AIが勝手に料理を選んでそのURLに飛んではならない。
    必ず検索結果ページに留まること。Youtubeの動画ページに移動することを禁止する。
    """

    # promptの定義
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])



    # {site}の定義
    # それぞれ特徴を持たせてクックパッド検索、マカロニ検索ができるよう
    # ここにも特徴的なキーワードを忍ばせる
    # streamlitでselectboxを意識して選択肢を考えておく



    messages = chat_prompt.format_prompt(site=site, material=material).to_messages()



    # OpenAI Chat Completion API の定義
    # 外部検索が目的なので、LangChainのAgentを使用する
    # gpt-4モデルを使いたいので、.envにOPENAI_API_KEYを記述
    # ツールは google-search を採用するため、.envにGOOGLE_API_KEY, GOOGLE_CSE_IDを追記

    # 練習の為、.envからAPIモデル名とtemperatureを読み込む実験。他に流用できる箇所があれば取り入れよう。

    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
    )

    tools = load_tools(["google-search"])


    # Agentを初期化して変数に入れる
    # AgentTypeはOPENAI_FUNCTIONSを採用

    agent_chain = initialize_agent(
            tools,
            chat,
            agent=AgentType.OPENAI_FUNCTIONS,
    )
    
    return agent_chain.run(messages)

# 初期設定
if material == "":
    exit()
if retrieve:
    result = retrieve_recipe(site, material)
    material = ""
else:
    exit()



# result = retrieve_recipe(site, material)


# デフォルトブラウザでURLを開く
url = result
webbrowser.open_new(url)



