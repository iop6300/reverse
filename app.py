import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import tempfile
import pandas as pd
import os

# --- [ROBOT-BASED LARGE-SCALE COMPOSITE AM] PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-REVERSE | Large-Scale Robot AM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS INJECTION ---
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("assets/style.css")

# --- SIDEBAR: INDUSTRIAL IDENTITY ---
with st.sidebar:
    st.markdown("<h1 style='color: #0052FF; font-size: 1.2rem; border: none; padding: 0;'>ROBOT-AM REVERSE</h1>", unsafe_allow_html=True)
    st.caption("v2.5.0 | Large-Scale Composite Edition")
    st.markdown("---")
    
    st.markdown("### 🛰️ SYSTEM STATUS")
    st.success("CORE ENGINE: ONLINE")
    st.info("ROBOT KINEMATICS: SYNC")
    
    st.markdown("---")
    st.markdown("### 📘 SYSTEM CONTEXT")
    st.markdown("""
    **Target Application:**
    - Robot-based Large-Scale AM
    - Carbon/Glass Fiber Composite
    - High-Precision Aerospace/Auto
    """)
    
    st.markdown("---")
    st.caption("© 2026 LARGE-SCALE AM R&D")

# --- MAIN INTERFACE HEADER ---
st.title("ROBOT-BASED COMPOSITE REVERSE-CALIBRATION")
st.markdown("<p style='color: #6B7280; margin-top: -1rem;'>Hybrid Error Compensation for Large-Scale Additive Manufacturing Structures</p>", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'calibration_constants' not in st.session_state:
    st.session_state['calibration_constants'] = None

# --- CORE UTILITIES ---
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
            st.error(f"DATA ERROR: {e}")
            return None, None
    return None, None

def gaussian_smooth_vectors(vertices, vectors, sigma=1.0):
    tree = cKDTree(vertices)
    radius = 3 * sigma
    indices_list = tree.query_ball_point(vertices, r=radius)
    smoothed_vectors = np.zeros_like(vectors)
    for i, indices in enumerate(indices_list):
        if not indices:
            smoothed_vectors[i] = vectors[i]
            continue
        weights = np.exp(-np.sum((vertices[indices] - vertices[i])**2, axis=1) / (2 * sigma**2))
        smoothed_vectors[i] = np.dot(weights, vectors[indices]) / np.sum(weights) if np.sum(weights) > 0 else vectors[i]
    return smoothed_vectors

# --- TABS ---
tab1, tab2 = st.tabs(["01 ANALYSIS", "02 DEPLOYMENT"])

# --- TAB 1: ANALYSIS ---
with tab1:
    col_l, col_r = st.columns([1, 2.2])
    
    with col_l:
        st.subheader("Source Input")
        cal_cad_file = st.file_uploader("CAD REFERENCE (ORIGINAL)", type=["stl", "obj"], key="cal_cad")
        cal_scan_file = st.file_uploader("SCAN DATA (MEASURED)", type=["stl", "obj"], key="cal_scan")
        
        st.divider()
        st.subheader("Compute Config")
        if st.button("RUN ENGINE", key="btn_cal"):
            if cal_cad_file and cal_scan_file:
                with st.spinner("CALIBRATING GEOMETRY..."):
                    cad_mesh, _ = load_mesh(cal_cad_file)
                    scan_mesh, _ = load_mesh(cal_scan_file)
                    
                    # --- [NEW] ADAPTIVE SUBDIVISION TO BRIDGE DENSITY GAP ---
                    # CAD 정점이 너무 적으면 스캔 데이터의 정밀도를 반영하기 위해 망 세분화 수행
                    if len(cad_mesh.vertices) < 10000:
                        st.toast("Low-res CAD detected. Increasing Mesh Density...", icon="🔍")
                        cad_mesh = cad_mesh.subdivide()
                    
                    cad_mesh.apply_translation(-cad_mesh.centroid)
                    scan_mesh.apply_translation(-scan_mesh.centroid)
                    
                    scale_factors = cad_mesh.extents / scan_mesh.extents
                    closest_points, _, _ = scan_mesh.nearest.on_surface(cad_mesh.vertices)
                    diff_vectors = closest_points - cad_mesh.vertices
                    
                    st.session_state['calibration_constants'] = scale_factors
                    st.session_state['cal_results'] = {
                        'vertices': cad_mesh.vertices, 'faces': cad_mesh.faces,
                        'normals': cad_mesh.vertex_normals, 'diff_vectors': diff_vectors,
                        'scale_factors': scale_factors
                    }
            else:
                st.warning("FILES MISSING")

    with col_r:
        if 'cal_results' in st.session_state and st.session_state['cal_results'] is not None:
            res = st.session_state['cal_results']
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("SCALE X", f"{res['scale_factors'][0]:.5f}", f"{(res['scale_factors'][0]-1)*100:+.3f}%")
            m2.metric("SCALE Y", f"{res['scale_factors'][1]:.5f}", f"{(res['scale_factors'][1]-1)*100:+.3f}%")
            m3.metric("SCALE Z", f"{res['scale_factors'][2]:.5f}", f"{(res['scale_factors'][2]-1)*100:+.3f}%")
            
            # 3D Viewport
            with st.expander("ALGORITHM CONTROLS", expanded=False):
                co1, co2 = st.columns(2)
                with co1:
                    use_smooth = st.checkbox("SMOOTHING", value=True)
                    sigma = st.slider("SIGMA (σ)", 0.1, 5.0, 1.2, 0.1, disabled=not use_smooth)
                with co2:
                    alpha = st.slider("ALPHA (α)", 0.0, 2.0, 1.0, 0.05)
            
            final_diff = gaussian_smooth_vectors(res['vertices'], res['diff_vectors'], sigma) if use_smooth else res['diff_vectors']
            dist = np.sum(final_diff * res['normals'], axis=1)
            max_dist = np.max(np.abs(dist)) or 0.1
            
            fig = go.Figure(data=[go.Mesh3d(
                x=res['vertices'][:,0], y=res['vertices'][:,1], z=res['vertices'][:,2],
                i=res['faces'][:,0], j=res['faces'][:,1], k=res['faces'][:,2],
                intensity=dist, colorscale='Portland', cmin=-max_dist, cmax=max_dist,
                reversescale=False,
                colorbar=dict(title="ERR (mm)", thickness=15, tickfont=dict(color="#111827")),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1)
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(gridcolor='#F3F4F6', zerolinecolor='#E5E7EB', backgroundcolor='white', tickfont=dict(color='#6B7280')),
                    yaxis=dict(gridcolor='#F3F4F6', zerolinecolor='#E5E7EB', backgroundcolor='white', tickfont=dict(color='#6B7280')),
                    zaxis=dict(gridcolor='#F3F4F6', zerolinecolor='#E5E7EB', backgroundcolor='white', tickfont=dict(color='#6B7280')),
                    bgcolor='white'
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(l=0, r=0, b=0, t=0), 
                height=650
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export
            st.markdown("### EXPORT ASSETS")
            ce1, ce2 = st.columns(2)
            with ce1:
                comp_mesh = trimesh.Trimesh(vertices=res['vertices'] - (final_diff * alpha), faces=res['faces'])
                tmp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                comp_mesh.export(tmp_f.name)
                with open(tmp_f.name, "rb") as f:
                    st.download_button("📥 DOWNLOAD COMPENSATED STL", f, "ai_compensated_light.stl", use_container_width=True)
            with ce2:
                st.download_button("📊 DOWNLOAD VECTOR DATA (CSV)", pd.DataFrame(final_diff).to_csv(index=False), "ai_vector_light.csv", use_container_width=True)
        else:
            st.info("AWAITING GEOMETRIC DATA FOR ANALYSIS")

# --- TAB 2: DEPLOYMENT ---
with tab2:
    if st.session_state['calibration_constants'] is None:
        st.warning("⚠️ CALIBRATION REQUIRED BEFORE DEPLOYMENT")
    else:
        v_col1, v_col2 = st.columns([1, 2.2])
        factors = st.session_state['calibration_constants']
        
        with v_col1:
            st.subheader("Mass Deployment")
            st.markdown(f"""
            **ACTIVE CALIBRATION PROFILE:**
            - X Factor: `{factors[0]:.4f}`
            - Y Factor: `{factors[1]:.4f}`
            - Z Factor: `{factors[2]:.4f}`
            """)
            val_cad_file = st.file_uploader("TARGET CAD MODEL", type=["stl", "obj"], key="val_cad")
            
            if val_cad_file:
                if st.button("PROCESS FOR DEPLOYMENT", key="btn_val"):
                    with st.spinner("SYNTHESIZING..."):
                        val_mesh, _ = load_mesh(val_cad_file)
                        val_mesh.vertices *= factors
                        st.session_state['val_result_mesh'] = val_mesh
        
        with v_col2:
            if 'val_result_mesh' in st.session_state:
                val_mesh = st.session_state['val_result_mesh']
                st.success("SYNTHESIS COMPLETE")
                
                fig_v = go.Figure(data=[go.Mesh3d(
                    x=val_mesh.vertices[:,0], y=val_mesh.vertices[:,1], z=val_mesh.vertices[:,2],
                    i=val_mesh.faces[:,0], j=val_mesh.faces[:,1], k=val_mesh.faces[:,2],
                    color='#0052FF', opacity=0.8,
                    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1)
                )])
                fig_v.update_layout(
                    scene=dict(
                        xaxis=dict(gridcolor='#F3F4F6', visible=True),
                        yaxis=dict(gridcolor='#F3F4F6', visible=True),
                        zaxis=dict(gridcolor='#F3F4F6', visible=True),
                        bgcolor='white'
                    ),
                    paper_bgcolor='white',
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=650
                )
                st.plotly_chart(fig_v, use_container_width=True)
                
                tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
                val_mesh.export(tmp_v.name)
                with open(tmp_v.name, "rb") as f:
                    st.download_button("📥 DOWNLOAD PRODUCTION STL", f, "ai_final_light.stl", use_container_width=True)
            else:
                st.info("AWAITING TARGET MODEL")

# --- FOOTER ---
st.markdown("---")
st.markdown("<center style='color: #9CA3AF; font-size: 0.75rem; letter-spacing: 0.05em;'>AI-REVERSE | SaaS INDUSTRIAL PROTOCOL</center>", unsafe_allow_html=True)
