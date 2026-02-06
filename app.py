import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import tempfile
import pandas as pd
import os

# 페이지 설정
st.set_page_config(page_title="3D 프린팅 역보정 툴", layout="wide")

st.title("🖨️ 3D 프린팅 역보정 및 검증 시스템")
st.markdown("""
이 시스템은 **캘리브레이션(Calibration)** 단계에서 보정 상수를 추출하고, 
**검증(Validation)** 단계에서 이를 설계에 적용하여 보정된 모델을 생성합니다.
""")

# 세션 상태 초기화 (보정 상수 저장용)
if 'calibration_constants' not in st.session_state:
    st.session_state['calibration_constants'] = None

# 유틸리티 함수: 메쉬 로드
def load_mesh(uploaded_file):
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            mesh = trimesh.load(tmp_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            return mesh, tmp_path
        except Exception as e:
            st.error(f"파일 로드 오류: {e}")
            return None, None
    return None, None

# 유틸리티 함수: 가우시안 스무딩 (Vector Field Smoothing)
def gaussian_smooth_vectors(vertices, vectors, sigma=1.0):
    """
    각 버텍스의 변위 벡터를 주변 이웃들과의 거리 기반 가우시안 가중치로 스무딩합니다.
    급격한 단면 변화나 노이즈로 인한 보정 값 튐 현상을 방지합니다.
    """
    tree = cKDTree(vertices)
    # 3 sigma 범위 내의 이웃만 탐색 (99.7% 커버)
    radius = 3 * sigma
    indices_list = tree.query_ball_point(vertices, r=radius)
    
    smoothed_vectors = np.zeros_like(vectors)
    
    # 성능을 위해 루프 최소화가 필요하지만, 가변 길이 이웃 처리를 위해 루프 사용
    # Streamlit progress bar와 함께 사용하는 것이 좋음
    for i, indices in enumerate(indices_list):
        if not indices:
            smoothed_vectors[i] = vectors[i]
            continue
            
        # 이웃들의 좌표와 벡터
        neighbors_pts = vertices[indices]
        neighbors_vecs = vectors[indices]
        
        # 현재 점과 이웃 간의 거리 제곱 계산
        dists_sq = np.sum((neighbors_pts - vertices[i])**2, axis=1)
        
        # Gaussian Kernel: exp(-d^2 / (2*sigma^2))
        weights = np.exp(-dists_sq / (2 * sigma**2))
        
        # 가중 평균 계산
        weighted_sum = np.dot(weights, neighbors_vecs)
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            smoothed_vectors[i] = weighted_sum / total_weight
        else:
            smoothed_vectors[i] = vectors[i]
            
    return smoothed_vectors

# 탭 구성
tab1, tab2 = st.tabs(["1️⃣ 캘리브레이션 (상수 추출)", "2️⃣ 검증 및 보정 설계 (상수 적용)"])

# --- 탭 1: 캘리브레이션 ---
with tab1:
    st.header("Step 1. 캘리브레이션 (보정 상수 추출)")
    st.markdown("CAD 원본과 실제 출력물(Scan)을 비교하여 **X, Y, Z 방향의 보정 상수(Scaling Factor)**를 산출합니다.")
    
    col1_up, col2_up = st.columns(2)
    with col1_up:
        cal_cad_file = st.file_uploader("CAD 원본 (Target)", type=["stl", "obj"], key="cal_cad")
    with col2_up:
        cal_scan_file = st.file_uploader("실제 스캔 데이터 (Scan)", type=["stl", "obj"], key="cal_scan")
        
    # 분석 버튼
    if cal_cad_file and cal_scan_file:
        if st.button("🔍 분석 및 상수 추출", key="btn_cal"):
            with st.spinner("형상 분석 및 보정 상수 계산 중..."):
                cad_mesh, _ = load_mesh(cal_cad_file)
                scan_mesh, _ = load_mesh(cal_scan_file)
                
                # 1. 중심 정렬 (Centering)
                cad_center = cad_mesh.centroid
                scan_center = scan_mesh.centroid
                scan_mesh.apply_translation(cad_center - scan_center)
                
                # 2. Bounding Box 기반 스케일링 팩터 계산
                cad_extents = cad_mesh.extents
                scan_extents = scan_mesh.extents
                scale_factors = cad_extents / scan_extents
                
                # 3. Raw Distance 계산
                closest_points, distances, _ = scan_mesh.nearest.on_surface(cad_mesh.vertices)
                diff_vectors = closest_points - cad_mesh.vertices # Raw Displacement
                
                # 결과 Session State에 저장
                st.session_state['calibration_constants'] = scale_factors
                st.session_state['cal_results'] = {
                    'vertices': cad_mesh.vertices,
                    'faces': cad_mesh.faces,
                    'normals': cad_mesh.vertex_normals,
                    'diff_vectors': diff_vectors,
                    'scale_factors': scale_factors
                }
                
                st.success("✅ 분석 완료! 아래에서 옵션을 조정하세요.")

    # 분석 결과가 있으면 표시 (슬라이더 조작 시에도 유지됨)
    if 'cal_results' in st.session_state and st.session_state['cal_results'] is not None:
        res = st.session_state['cal_results']
        scale_factors = res['scale_factors']
        
        # 결과 표시
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("X축 보정 상수", f"{scale_factors[0]:.4f}", f"{(scale_factors[0]-1)*100:.2f}%")
        col_res2.metric("Y축 보정 상수", f"{scale_factors[1]:.4f}", f"{(scale_factors[1]-1)*100:.2f}%")
        col_res3.metric("Z축 보정 상수", f"{scale_factors[2]:.4f}", f"{(scale_factors[2]-1)*100:.2f}%")
        
        st.info("이 상수는 '검증 및 보정 설계' 탭에서 자동으로 사용됩니다.")
        
        # --- 상세 오차 분석 및 스무딩 로직 ---
        st.divider()
        st.subheader("상세 오차 분석 및 보정 맵 생성")
        
        # 저장된 데이터 불러오기
        vertices = res['vertices']
        normals = res['normals']
        diff_vectors = res['diff_vectors']
        faces = res['faces']

        # 2. 가우시안 스무딩 옵션
        use_smoothing = st.checkbox("가우시안 스무딩 적용 (노이즈/튀는 값 제거)", value=True)
        sigma_val = 1.0
        
        final_diff_vectors = diff_vectors # 기본값
        
        if use_smoothing:
            sigma_val = st.slider("Smoothing Sigma (영향 반경 조절)", 0.1, 5.0, 1.0, 0.1, help="값이 클수록 더 넓은 영역을 평균화하여 부드럽게 만듭니다.")
            # 스무딩 연산은 무거울 수 있으므로 캐싱하면 좋지만, 파라미터가 바뀌므로 매번 계산
            # (최적화를 위해선 hash_func 등을 써야하지만 여기선 단순 구현)
            with st.spinner(f"벡터 필드 스무딩 중... (Sigma={sigma_val})"):
                final_diff_vectors = gaussian_smooth_vectors(vertices, diff_vectors, sigma=sigma_val)
        
        # 3. 시각화 (Signed Distance)
        # 전체 버텍스에 대해 오차 계산 (Mesh3d에 색상을 입히기 위함)
        viz_signed_dist = np.sum(final_diff_vectors * normals, axis=1)
        
        max_abs_dist = np.max(np.abs(viz_signed_dist))
        if max_abs_dist == 0: max_abs_dist = 0.1
        
        custom_colorscale = [[0.0, "blue"], [0.5, "green"], [1.0, "red"]]
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                intensity=viz_signed_dist,
                colorscale=custom_colorscale,
                cmin=-max_abs_dist, cmax=max_abs_dist,
                colorbar=dict(title="오차 (mm)"), showscale=True,
                name='오차 분포'
            )
        ])
        fig.update_layout(scene=dict(aspectmode='data'), title=f"오차 분포 (Smoothing: {'ON' if use_smoothing else 'OFF'})")
        st.plotly_chart(fig, use_container_width=True)

        # 4. 보정된 모델 생성 및 다운로드
        alpha = st.slider("보정 계수 (Alpha)", 0.5, 1.5, 1.0, 0.1)
        
        # 보정: CAD - (Error Vector * Alpha)
        compensated_vertices = vertices - (final_diff_vectors * alpha)
        compensated_mesh = trimesh.Trimesh(vertices=compensated_vertices, faces=faces)
        
        tmp_export = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        compensated_mesh.export(tmp_export.name)
        
        with open(tmp_export.name, "rb") as f:
            st.download_button("📥 보정된 STL 파일 다운로드", f, "compensated_model.stl", "model/stl")

        # 5. CSV 다운로드
        df_calib = pd.DataFrame(final_diff_vectors, columns=["Diff_X", "Diff_Y", "Diff_Z"])
        df_calib["Calib_X"] = -final_diff_vectors[:,0]
        df_calib["Calib_Y"] = -final_diff_vectors[:,1]
        df_calib["Calib_Z"] = -final_diff_vectors[:,2]
        csv = df_calib.to_csv(index=False).encode('utf-8')
        st.download_button("📊 보정 데이터 (CSV) 다운로드", csv, "calibration_data.csv", "text/csv")

# --- 탭 2: 검증 및 보정 ---
with tab2:
    st.header("Step 2. 검증 및 보정 설계")
    
    # 저장된 상수 확인
    if st.session_state['calibration_constants'] is None:
        st.warning("⚠️ 먼저 '캘리브레이션' 탭에서 보정 상수를 추출해주세요.")
    else:
        factors = st.session_state['calibration_constants']
        st.markdown(f"""
        **현재 적용된 보정 상수:**
        - **X:** {factors[0]:.4f} | **Y:** {factors[1]:.4f} | **Z:** {factors[2]:.4f}
        """)
        
        st.markdown("보정할 **새로운 CAD 파일**을 업로드하면 위 상수를 적용하여 역보정 모델을 생성합니다.")
        
        val_cad_file = st.file_uploader("설계 CAD 파일 (Design)", type=["stl", "obj"], key="val_cad")
        val_sim_file = st.file_uploader("시뮬레이션/예측 형상 (선택 사항)", type=["stl", "obj"], key="val_sim")
        
        if val_cad_file:
            if st.button("🛠️ 보정 모델 생성 및 검증", key="btn_val"):
                cad_mesh_val, _ = load_mesh(val_cad_file)
                
                # 보정 적용 (Scaling)
                # Vertex 좌표에 스케일 팩터 곱하기
                # 중심 기준으로 스케일링하기 위해 중심 이동 -> 스케일 -> 원복
                center = cad_mesh_val.centroid
                cad_mesh_val.vertices -= center
                cad_mesh_val.vertices *= factors
                cad_mesh_val.vertices += center
                
                st.success("✅ 역보정 적용 완료!")
                
                # 시각화 비교
                viz_data = [
                    go.Mesh3d(
                        x=cad_mesh_val.vertices[:,0], y=cad_mesh_val.vertices[:,1], z=cad_mesh_val.vertices[:,2],
                        i=cad_mesh_val.faces[:,0], j=cad_mesh_val.faces[:,1], k=cad_mesh_val.faces[:,2],
                        opacity=0.5, color='green', name='보정된 모델'
                    )
                ]
                
                # 시뮬레이션 파일이 있다면 함께 표시
                if val_sim_file:
                    sim_mesh, _ = load_mesh(val_sim_file)
                    viz_data.append(
                        go.Mesh3d(
                            x=sim_mesh.vertices[:,0], y=sim_mesh.vertices[:,1], z=sim_mesh.vertices[:,2],
                            i=sim_mesh.faces[:,0], j=sim_mesh.faces[:,1], k=sim_mesh.faces[:,2],
                            opacity=0.3, color='red', name='시뮬레이션 예측'
                        )
                    )
                    st.info("초록색: 보정된 모델, 빨간색: 시뮬레이션 예측")
                
                fig_val = go.Figure(data=viz_data)
                fig_val.update_layout(scene=dict(aspectmode='data'), title="보정 결과 확인")
                st.plotly_chart(fig_val, use_container_width=True)
                
                # 다운로드
                tmp_export_val = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                cad_mesh_val.export(tmp_export_val.name)
                with open(tmp_export_val.name, "rb") as f:
                    st.download_button(
                        label="📥 보정된 CAD 파일 다운로드 (Compensated STL)",
                        data=f,
                        file_name="compensated_design.stl",
                        mime="model/stl"
                    )
