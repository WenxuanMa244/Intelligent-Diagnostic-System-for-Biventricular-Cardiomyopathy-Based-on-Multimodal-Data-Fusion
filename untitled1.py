# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 00:57:48 2025

@author: 32572
"""
import streamlit as st
import ollama

def main():
    st.title("仲景-医学垂直大模型")

    # 设置用户输入框
    user_input = st.text_area("您想问什么？", "")

    # 当使用者按下送出按钮后的处理
    if st.button("送出"):
        if user_input:
            # 使用ollama模型进行对话
            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': user_input}])
            
            # 显示回答
            st.text("回答：")
            st.write(response['message']['content'])
        else:
            st.warning("请输入问题！")

if __name__ == "__main__":
    main()
