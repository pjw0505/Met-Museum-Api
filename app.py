# Met-Museum-Api
import streamlit as st
import replicate
import time
import os

# 페이지 설정
st.set_page_config(page_title="AI 이미지 생성기", layout="wide")
st.title("🎨 Streamlit AI 이미지 생성기")

# Streamlit Community Cloud에 배포하는 경우
# Replicate API 키는 .streamlit/secrets.toml 파일에 저장해야 합니다.
# [replicate]
# api_token = "YOUR_REPLICATE_API_TOKEN"
try:
    REPLICATE_API_TOKEN = st.secrets["replicate"]["api_token"]
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
except:
    st.error("Replicate API 키를 설정해주세요. (secrets.toml 또는 환경 변수)")
    REPLICATE_API_TOKEN = None

# UI 구성
prompt = st.text_area("✨ 생성하고 싶은 이미지에 대한 설명을 입력하세요:", "A beautiful watercolor painting of a futuristic city at sunset, highly detailed.")

with st.sidebar:
    st.header("설정")
    width = st.selectbox("이미지 가로 크기", [512, 768, 1024], index=2)
    height = st.selectbox("이미지 세로 크기", [512, 768, 1024], index=2)
    
    # Replicate 모델을 위한 매개변수
    num_outputs = st.slider("생성할 이미지 수", 1, 4, 1)
    
    st.markdown("---")
    st.markdown("본 앱은 **Replicate API**를 사용합니다.")


if st.button("이미지 생성", use_container_width=True) and REPLICATE_API_TOKEN:
    
    if not prompt:
        st.warning("설명을 입력해주세요!")
    else:
        # Replicate API 호출
        with st.spinner('이미지를 생성 중입니다... 잠시만 기다려주세요.'):
            try:
                # 사용 모델: stability-ai/sdxl
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_outputs": num_outputs,
                        "negative_prompt": "low quality, worst quality, bad anatomy, deformed"
                    }
                )
                
                # 결과 표시
                st.success("이미지 생성 완료!")
                
                if output:
                    cols = st.columns(num_outputs)
                    for i, image_url in enumerate(output):
                        with cols[i]:
                            st.image(image_url, caption=f"결과 {i+1}", use_column_width="always")
                else:
                    st.error("이미지 생성 결과가 없습니다.")
                    
            except Exception as e:
                st.error(f"이미지 생성 중 오류 발생: {e}")
