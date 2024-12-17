import streamlit as st
import requests
from datetime import timedelta, datetime

# 配置 FastAPI 后端 URL
backend_url = "http://127.0.0.1:8000/predict"

st.title("股票预测服务")

# 输入股票信息
stock_names = ["600519.SS", "300750.sz", "600036.ss", "601318.ss", "000002.SZ", "9988.HK", "0700.HK", "000651.SZ", "600276.SS", "3690.HK"]
stock_name = st.selectbox("选择股票代码", stock_names)
start_date = st.date_input("开始日期")
end_date = st.date_input("结束日期")

if st.button("预测"):
    # 调用后端 API
    response = requests.post(backend_url, json={
        "stock_name": stock_name,
        "start_date": str(start_date),
        "end_date": str(end_date)
    })
    if response.status_code == 200:
        result = response.json()["predicted_prices"]
        st.write(f"预测结果：")
        day_1 = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        day_2 = (end_date + timedelta(days=2)).strftime("%Y-%m-%d")
        day_3 = (end_date + timedelta(days=3)).strftime("%Y-%m-%d")
        st.write(f"{day_1}: {result['Day 1']}")
        st.write(f"{day_2}: {result['Day 2']}")
        st.write(f"{day_3}: {result['Day 3']}")
        st.write(f"实际结果：")
        day_1 = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        day_2 = (end_date + timedelta(days=2)).strftime("%Y-%m-%d")
        day_3 = (end_date + timedelta(days=3)).strftime("%Y-%m-%d")
        st.write(f"{day_1}: {result['Day 1']}")
        st.write(f"{day_2}: {result['Day 2']}")
        st.write(f"{day_3}: {result['Day 3']}")
    else:
        st.error("预测失败，请稍后再试！")
