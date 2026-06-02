from openai import OpenAI
client = OpenAI(
    api_key="sk-ERkXc0xG66roetUpHXnmsGQYO7Y4dkyic09emAO0DSDjw0Zh",
    base_url="https://ai.nengyongai.cn/v1"
)

response = client.chat.completions.create(
    messages=[
        # 把用户提示词传进来content
        {'role': 'user', 'content': "鲁迅为什么打周树人？"},
    ],
    model='gpt-4o-audio-preview',  # 上面写了可以调用的模型
    stream=True  # 一定要设置Tru
)
for chunk in response:
    print(chunk.choices[0].content, end="", flush=True)