import streamlit as st
import torch
from diffusers import StableDiffusionPipeline # StableDiffusionPipeline 사용

# --- 모델 로드 (캐싱 필수: 앱이 리로드 되어도 모델을 다시 불러오지 않도록 함) ---
@st.cache_resource
def load_model():
    # 사용하려는 Stable Diffusion 모델 ID 지정 (Hugging Face ID)
    model_id = "runwayml/stable-diffusion-v1-5" 
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner(f"모델을 {device}에 로드 중입니다... (최초 실행 시 시간 소요)"):
        # 파이프라인 로드
        if device == "cuda":
            # GPU 사용 시 메모리 절약을 위해 float16 사용
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.to(device)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(model_id)
            pipeline.to(device)
            
    return pipeline

st.title("✨ 로컬 $\text{Stable Diffusion}$ 이미지 생성기")

# 모델 로드
pipeline = load_model()

# --- UI 입력 ---
prompt = st.text_area(
    "생성하고 싶은 이미지 설명을 입력하세요:", 
    "A photorealistic image of a cat wearing a spacesuit, digital art."
)
negative_prompt = st.text_input(
    "제외하고 싶은 요소 (Negative Prompt):", 
    "low quality, worst quality, bad anatomy, deformed"
)

# 사이드바 설정
with st.sidebar:
    st.header("생성 설정")
    num_inference_steps = st.slider("Step 수", 10, 100, 50)
    guidance_scale = st.slider("Guidance Scale (CFG)", 1.0, 20.0, 7.5)
    seed = st.number_input("Seed 값 (랜덤성을 위해 비워두세요)", value=None, format="%d")

# 시드 설정 (재현성을 위해 필요)
generator = torch.Generator(pipeline.device).manual_seed(seed) if seed is not None else None

if st.button("이미지 생성", use_container_width=True):
    if not prompt:
        st.warning("설명을 입력해야 합니다.")
    else:
        with st.spinner("이미지를 생성 중입니다..."):
            # 이미지 생성 호출
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            # 결과 표시
            st.image(image, caption="생성된 이미지", use_column_width=True)
            st.success("이미지 생성 완료!")
