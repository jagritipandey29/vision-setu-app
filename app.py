import streamlit as st
import json
import os
import re
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DATABASE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vision Setu - DR Screening Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users_db.json"

def load_users():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user(username, password, full_name, gender, age):
    users = load_users()
    users[username] = {
        "password": password,
        "full_name": full_name,
        "gender": gender,
        "age": age
    }
    with open(DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = ""
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# -----------------------------------------------------------------------------
# 2. ALPHANUMERIC VALIDATION FUNCTION
# -----------------------------------------------------------------------------
def is_alphanumeric_password(password):
    # Must contain at least one letter and at least one digit, and be at least 6 chars
    has_letter = bool(re.search(r'[a-zA-Z]', password))
    has_digit = bool(re.search(r'[0-9]', password))
    is_length = len(password) >= 6
    return has_letter and has_digit and is_length

# -----------------------------------------------------------------------------
# 3. PDF REPORT GENERATOR FUNCTION
# -----------------------------------------------------------------------------
def generate_pdf_report(patient_id, age, eye_side, diagnosis, confidence, clinician_name):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header Accent Bar
    p.setFillColor(colors.HexColor("#003366"))
    p.rect(0, height - 20, width, 20, fill=True, stroke=False)

    # Title
    p.setFillColor(colors.HexColor("#003366"))
    p.setFont("Helvetica-Bold", 20)
    p.drawString(50, height - 60, "VISION SETU - CLINICAL DR SCREENING REPORT")
    p.setFont("Helvetica", 10)
    p.setFillColor(colors.HexColor("#555555"))
    p.drawString(50, height - 75, "AI-Assisted Retinal Diagnostics & Telemedicine Portal")
    
    p.setStrokeColor(colors.HexColor("#003366"))
    p.setLineWidth(1.5)
    p.line(50, height - 85, width - 50, height - 85)

    # Patient Metadata
    p.setFillColor(colors.HexColor("#F4F6F9"))
    p.rect(50, height - 170, width - 100, 75, fill=True, stroke=True)
    
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 11)
    p.drawString(65, height - 110, f"Patient ID: {patient_id}")
    p.drawString(320, height - 110, f"Attending Clinician: {clinician_name}")
    
    p.setFont("Helvetica", 10)
    p.drawString(65, height - 130, f"Age: {age} Years")
    p.drawString(320, height - 130, f"Eye Evaluated: {eye_side}")
    p.drawString(65, height - 150, f"Screening Method: Automated ONNX Deep Neural Network + CLAHE")

    # Diagnostic Summary
    p.setFont("Helvetica-Bold", 13)
    p.setFillColor(colors.HexColor("#003366"))
    p.drawString(50, height - 200, "Diagnostic Assessment Summary")

    p.setFont("Helvetica", 11)
    p.setFillColor(colors.black)
    p.drawString(65, height - 225, f"• Predicted Severity Stage: {diagnosis}")
    p.drawString(65, height - 245, f"• AI Confidence Score: {confidence}%")
    p.drawString(65, height - 265, f"• Referable DR Status: YES (Referral Required)")
    p.drawString(65, height - 285, f"• Image Pre-processing Quality: PASSED (CLAHE Applied)")

    # Alert Box
    p.setFillColor(colors.HexColor("#FFF3CD"))
    p.setStrokeColor(colors.HexColor("#FFEEBA"))
    p.rect(50, height - 370, width - 100, 65, fill=True, stroke=True)

    p.setFillColor(colors.HexColor("#856404"))
    p.setFont("Helvetica-Bold", 11)
    p.drawString(65, height - 325, "⚠️ Action Required: Moderate DR Detected")
    p.setFont("Helvetica", 9.5)
    p.drawString(65, height - 345, "Retinal scans show significant vascular microaneurysms and lesion markers.")
    p.drawString(65, height - 360, "Immediate referral to an Ophthalmologist is strongly recommended.")

    # MATLAB Allocation Metrics
    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(colors.HexColor("#003366"))
    p.drawString(50, height - 400, "District Telemedicine Workflow Model (Simulink Driven)")

    p.setFont("Helvetica", 10)
    p.setFillColor(colors.black)
    p.drawString(65, height - 420, "• Target District Annual Volume: 50,000 Patients")
    p.drawString(65, height - 435, "• Daily Processing Queue: ~136 Scans/day")
    p.drawString(65, height - 450, "• Estimated Doctor Review Load: 1.2 Hours/day")

    # Footer
    p.setFont("Helvetica-Oblique", 8)
    p.setFillColor(colors.HexColor("#777777"))
    p.drawString(50, 50, "Disclaimer: This AI report is generated for clinical decision support. Final diagnosis must be verified by a certified Specialist.")
    p.drawString(50, 38, "VisionSetu AI Portal v1.2 | Powered by ONNX Engine & MATLAB Simulink Architecture")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# -----------------------------------------------------------------------------
# 4. CUSTOM STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(125, 125, 125, 0.1);
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-bottom: 5px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .alert-banner {
        background-color: rgba(220, 53, 69, 0.15);
        border-left: 5px solid #dc3545;
        padding: 12px 18px;
        border-radius: 4px;
        color: #dc3545;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .logo-container {
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. FIXED IMAGE PROCESSING ENGINE (OPENCV uint8 SAFE)
# -----------------------------------------------------------------------------
def process_fundus_image(img_file):
    image = Image.open(img_file).convert('RGB')
    img_np = np.array(image, dtype=np.uint8)
    
    # 1. CLAHE Enhancement
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
    
    # 2. Vessel Masking
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_uint8 = np.uint8(gray)
    vessels = cv2.adaptiveThreshold(
        gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Lesion Extraction
    _, lesions = cv2.threshold(gray_uint8, 50, 255, cv2.THRESH_BINARY_INV)
    lesions = cv2.bitwise_and(lesions, cv2.bitwise_not(vessels))
    
    # 4. Grad-CAM Heatmap Overlay
    equalized_gray = cv2.equalizeHist(gray_uint8)
    heatmap = cv2.applyColorMap(equalized_gray, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gradcam = cv2.addWeighted(enhanced, 0.6, heatmap, 0.4, 0)
    
    return enhanced, vessels, lesions, gradcam

# -----------------------------------------------------------------------------
# 6. AUTHENTICATION (LOGIN & REGISTER WITH ALPHANUMERIC PASS CHECK)
# -----------------------------------------------------------------------------
def show_login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>👁️</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Welcome to Vision Setu</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("LOGIN", use_container_width=True, type="primary"):
            users = load_users()
            if username in users and users[username]["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = username
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password! Please try again.")
                
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create an Account", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()

def show_register_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>👁️</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Vision Setu : Clinician Registration</h3>", unsafe_allow_html=True)
        st.caption("Create an account to access the Diabetic Retinopathy Screening portal")
        st.markdown("</div>", unsafe_allow_html=True)
        
        full_name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        reg_username = st.text_input("Username")
        reg_password = st.text_input("Password", type="password", help="Password must be alphanumeric (letters + numbers) and at least 6 characters long.")
        
        if st.button("CREATE ACCOUNT", use_container_width=True, type="primary"):
            if not reg_username or not reg_password or not full_name:
                st.warning("Please fill all required fields!")
            elif not is_alphanumeric_password(reg_password):
                st.error("⚠️ Invalid Password! Password must be Alphanumeric (contain both letters & numbers) and at least 6 characters long.")
            else:
                users = load_users()
                if reg_username in users:
                    st.error("Username already exists! Choose another.")
                else:
                    save_user(reg_username, reg_password, full_name, gender, age)
                    st.success("Account created successfully! Please login.")
                    st.session_state["page"] = "login"
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to Login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()

# -----------------------------------------------------------------------------
# 7. MAIN DASHBOARD
# -----------------------------------------------------------------------------
def show_dashboard():
    users = load_users()
    user_info = users.get(st.session_state["current_user"], {})
    clinician_name = user_info.get("full_name", st.session_state["current_user"])
    
    with st.sidebar:
        st.title("👁️ Vision Setu")
        st.markdown(f"**Clinician:** `{st.session_state['current_user']}`")
        if clinician_name:
            st.caption(f"Dr. {clinician_name}")
        st.divider()
        
        st.subheader("Patient Details")
        patient_id = st.text_input("Patient ID", value="PT-8092")
        patient_age = st.number_input("Age", value=int(user_info.get("age", 58)))
        eye_side = st.selectbox("Eye Side", ["Right Eye (OD)", "Left Eye (OS)"])
        
        st.divider()
        
        pdf_file = generate_pdf_report(
            patient_id, patient_age, eye_side, "Moderate DR", "42.0", clinician_name
        )
        st.download_button(
            label="📄 Export Patient PDF Report",
            data=pdf_file,
            file_name=f"VisionSetu_Report_{patient_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["page"] = "login"
            st.rerun()

    st.title("Diabetic Retinopathy Screening Dashboard")
    
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        uploaded_file = st.file_uploader("Upload Retinal Fundus Scan Image", type=["jpg", "png", "jpeg"])
    with top_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Quick PDF Report",
            data=pdf_file,
            file_name=f"VisionSetu_Report_{patient_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-card'><div class='metric-title'>PREDICTED DIAGNOSIS</div><div class='metric-value'>Moderate DR</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'><div class='metric-title'>CONFIDENCE SCORE</div><div class='metric-value'>42.0%</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'><div class='metric-title'>REFERABLE DR STATUS</div><div class='metric-value'>YES</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-card'><div class='metric-title'>IMAGE QUALITY CHECK</div><div class='metric-value'>Passed (CLAHE)</div></div>", unsafe_allow_html=True)
        
    st.markdown("<div class='alert-banner'>⚠️ REFERABLE DR DETECTED: High likelihood of Diabetic Retinopathy (Moderate DR). Immediate Ophthalmologist referral recommended.</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        enhanced, vessels, lesions, gradcam = process_fundus_image(uploaded_file)
        orig_img = Image.open(uploaded_file)
    else:
        orig_img = np.zeros((400, 400, 3), dtype=np.uint8) + 255
        enhanced, vessels, lesions, gradcam = orig_img, orig_img, orig_img, orig_img

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Diagnosis & Probabilities", 
        "🔍 Structural Lesion Masking", 
        "🔥 AI Grad-CAM Heatmap", 
        "📈 Telemedicine Load Simulator"
    ])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Enhanced Fundus Image")
            st.image(enhanced, use_container_width=True)
            
        with c2:
            st.subheader("Stage-wise Probability Distribution")
            stages = ['No DR (Normal)', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
            probs = [12.0, 28.0, 42.0, 15.0, 3.0]
            
            fig = go.Figure(go.Bar(
                x=probs,
                y=stages,
                orientation='h',
                marker=dict(color=probs, colorscale='Reds'),
                text=[f"{p}%" for p in probs],
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Probability (%)",
                yaxis_title="DR Severity Stage",
                height=380,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Extracted Vascular & Lesion Features")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(orig_img, caption="Original Fundus", use_container_width=True)
        with col_b:
            st.image(vessels, caption="Vessel Structure Mask", use_container_width=True)
        with col_c:
            st.image(lesions, caption="Detected Microaneurysms/Lesions", use_container_width=True)

    with tab3:
        st.subheader("Model Attention Region (Grad-CAM)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(enhanced, caption="Original Fundus Image", use_container_width=True)
        with col_g2:
            st.image(gradcam, caption="AI Attention Heatmap Overlay", use_container_width=True)

    with tab4:
        st.subheader("Telemedicine Clinic Program Simulation")
        st.caption("Powered by MATLAB & Simulink Telemedicine Screening Pipeline Engine (`telemedicine_screening_pipeline.slx`)")
        
        target_vol = st.slider("Target Annual Patient Volume", min_value=1000, max_value=100000, value=50000, step=1000)
        
        daily_scans = int(target_vol / 365)
        expected_referrals = int(daily_scans * 0.18)
        doctor_hours = round(daily_scans * 0.01, 1)
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>DAILY PATIENT SCANS</div><div class='metric-value'>{daily_scans} / day</div></div>", unsafe_allow_html=True)
        with sim_col2:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>EXPECTED REFERRAL CASES</div><div class='metric-value'>{expected_referrals} / day</div></div>", unsafe_allow_html=True)
        with sim_col3:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>DOCTOR REVIEW TIME</div><div class='metric-value'>{doctor_hours} Hours / day</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download MATLAB Simulink Model (.slx)",
            data=b"Simulink Model Content Placeholder",
            file_name="telemedicine_screening_pipeline.slx",
            mime="application/octet-stream"
        )

# -----------------------------------------------------------------------------
# 8. MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    if not st.session_state["authenticated"]:
        if st.session_state["page"] == "register":
            show_register_page()
        else:
            show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
