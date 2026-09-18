import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
    background-color: #fafafa;
}
.metric-title {
    font-size: 14px;
    color: #666;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
}
.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}
.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease that commonly causes dark olive or brown lesions on apple leaves and fruit.",
        "treatment": "Remove infected leaves and fallen plant material. Improve air circulation and use suitable fungicide management when recommended.",
        "prevention": "Maintain good spacing, prune for airflow, remove fallen infected leaves, and avoid prolonged leaf wetness."
    },

    "Apple___Black_rot": {
        "description": "Black rot is a fungal disease that can cause brown or purple leaf spots and dark lesions.",
        "treatment": "Remove infected plant material and maintain orchard sanitation. Fungicide management may be used when appropriate.",
        "prevention": "Remove dead or infected branches, maintain airflow, and avoid leaving infected fruit or plant debris."
    },

    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is a fungal disease that produces yellow-orange spots on apple leaves.",
        "treatment": "Remove heavily infected material and use appropriate fungicide management when recommended.",
        "prevention": "Maintain good airflow and manage nearby alternate hosts when practical."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew appears as a white powder-like growth on leaf surfaces.",
        "treatment": "Remove severely affected leaves and improve airflow. Appropriate fungicide management may be required.",
        "prevention": "Avoid overcrowding, maintain good air circulation, and monitor new growth regularly."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is a fungal disease producing elongated gray or brown lesions on corn leaves.",
        "treatment": "Remove infected plant residue where practical and use appropriate disease-management practices.",
        "prevention": "Use resistant varieties when available, maintain field sanitation, and avoid excessive plant density."
    },

    "Corn_(maize)___Common_rust_": {
        "description": "Common rust produces reddish-brown rust-colored pustules on corn leaves.",
        "treatment": "Monitor disease development and use appropriate fungicide management when necessary.",
        "prevention": "Use resistant varieties where available and maintain healthy crop growth."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight causes long, gray-green or brown lesions on corn leaves.",
        "treatment": "Remove infected crop residue where practical and use suitable disease-management practices.",
        "prevention": "Use resistant varieties, maintain crop rotation, and improve field sanitation."
    },

    "Grape___Black_rot": {
        "description": "Grape black rot is a fungal disease that causes brown lesions and can affect fruit.",
        "treatment": "Remove infected leaves and fruit and maintain vineyard sanitation. Appropriate fungicide management may be needed.",
        "prevention": "Improve airflow, remove infected plant material, and avoid excessive moisture on foliage."
    },

    "Grape___Esca_(Black_Measles)": {
        "description": "Esca is a grapevine disease associated with leaf symptoms and deterioration of vine tissues.",
        "treatment": "Remove severely affected plant material and maintain vineyard sanitation. Consult local plant-health guidance for management.",
        "prevention": "Use healthy planting material, minimize unnecessary wounds, and maintain good vineyard hygiene."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Grape leaf blight causes dark spots and lesions on leaves.",
        "treatment": "Remove infected leaves and improve canopy airflow. Appropriate fungicide management may be considered.",
        "prevention": "Maintain spacing, improve airflow, and remove infected plant debris."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Citrus greening is a serious citrus disease that can cause blotchy leaves and poor plant development.",
        "treatment": "Remove severely affected plants when recommended and control the insect vectors responsible for spreading the disease.",
        "prevention": "Use healthy planting material and monitor and manage insect vectors."
    },

    "Peach___Bacterial_spot": {
        "description": "Peach bacterial spot causes small dark lesions on leaves and fruit.",
        "treatment": "Remove severely affected plant material and maintain good orchard sanitation. Follow local recommendations for bacterial disease management.",
        "prevention": "Improve airflow, avoid unnecessary leaf wetness, and use healthy planting material."
    },

    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark spots on pepper leaves and may affect fruit.",
        "treatment": "Remove infected plant material and avoid working with wet plants. Use appropriate bacterial disease management where recommended.",
        "prevention": "Use clean seeds or seedlings, maintain spacing, and avoid overhead irrigation."
    },

    "Potato___Early_blight": {
        "description": "Early blight causes dark circular lesions on potato leaves, often with concentric rings.",
        "treatment": "Remove infected plant material and maintain crop sanitation. Appropriate fungicide management may be used.",
        "prevention": "Use crop rotation, maintain good plant nutrition, avoid prolonged leaf wetness, and remove infected debris."
    },

    "Potato___Late_blight": {
        "description": "Late blight is a destructive disease that causes dark lesions and rapid leaf damage.",
        "treatment": "Remove severely infected material and seek timely disease-management guidance. Appropriate fungicide management may be required.",
        "prevention": "Use healthy seed material, improve airflow, avoid prolonged leaf wetness, and monitor the crop frequently."
    },

    "Squash___Powdery_mildew": {
        "description": "Powdery mildew creates white powder-like patches on squash leaves.",
        "treatment": "Remove severely affected leaves and improve airflow. Suitable fungicide management may be considered.",
        "prevention": "Provide adequate spacing, avoid excessive humidity, and maintain good airflow."
    },

    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch produces dark or reddish-purple spots that can expand across strawberry leaves.",
        "treatment": "Remove infected leaves and maintain plant sanitation. Appropriate disease-management practices may be used.",
        "prevention": "Improve airflow, avoid prolonged leaf wetness, and remove infected plant debris."
    },

    "Tomato___Bacterial_spot": {
        "description": "Tomato bacterial spot causes small dark spots on leaves and can also affect fruit.",
        "treatment": "Remove infected plant material and avoid overhead irrigation. Follow suitable bacterial disease-management practices.",
        "prevention": "Use clean planting material, provide good spacing, and avoid handling wet plants."
    },

    "Tomato___Early_blight": {
        "description": "Tomato early blight causes dark brown lesions, often with concentric ring patterns, on older leaves.",
        "treatment": "Remove affected leaves and maintain plant sanitation. Appropriate fungicide management may be considered.",
        "prevention": "Use crop rotation, improve airflow, avoid overhead watering, and remove infected debris."
    },

    "Tomato___Late_blight": {
        "description": "Tomato late blight can cause dark, water-soaked lesions and rapid plant damage.",
        "treatment": "Remove infected plant material and seek timely disease-management guidance. Appropriate fungicide management may be necessary.",
        "prevention": "Improve airflow, avoid prolonged leaf wetness, and monitor plants regularly."
    },

    "Tomato___Leaf_Mold": {
        "description": "Tomato leaf mold commonly produces yellow areas on the upper leaf surface and fungal growth underneath.",
        "treatment": "Remove affected leaves and improve ventilation. Appropriate fungicide management may be considered.",
        "prevention": "Reduce humidity, improve airflow, avoid wetting foliage, and provide adequate plant spacing."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes numerous small circular spots, often with darker borders, on tomato leaves.",
        "treatment": "Remove infected leaves and plant debris. Suitable fungicide management may be used when recommended.",
        "prevention": "Avoid overhead irrigation, improve airflow, and maintain good garden sanitation."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mites are tiny pests that cause stippling, yellowing, and weakening of leaves.",
        "treatment": "Remove heavily affected leaves and use suitable pest-management practices. Encourage beneficial insects where practical.",
        "prevention": "Monitor plants regularly, reduce excessive dust and plant stress, and maintain suitable growing conditions."
    },

    "Tomato___Target_Spot": {
        "description": "Target spot is a fungal disease that produces circular brown lesions with target-like patterns.",
        "treatment": "Remove affected plant material and improve airflow. Appropriate fungicide management may be considered.",
        "prevention": "Avoid prolonged leaf wetness, maintain spacing, and remove infected debris."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato yellow leaf curl virus can cause yellowing, curling, and stunted growth.",
        "treatment": "Remove severely infected plants where recommended and manage whitefly vectors.",
        "prevention": "Use healthy seedlings, monitor whiteflies, and maintain good field sanitation."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus can cause mottled or mosaic patterns on leaves and reduced plant growth.",
        "treatment": "Remove infected plants and sanitize tools and hands to reduce mechanical spread.",
        "prevention": "Use clean planting material, disinfect tools, and avoid handling plants when they are wet."
    }
}


HEALTHY_INFO = {
    "description": "The leaf looks healthy based on the model prediction.",
    "treatment": "No treatment is required based on this prediction.",
    "prevention": "Continue regular watering, provide adequate sunlight and spacing, maintain good airflow, and inspect the plant regularly."
}


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model.")
    st.code(str(e))
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pretty(name):
    name = name.replace("___", " - ")
    name = name.replace("_", " ")
    name = name.replace("(", "")
    name = name.replace(")", "")
    return name


def get_plant_name(class_name):
    return class_name.split("___")[0].replace("_", " ")


def is_healthy(class_name):
    return "healthy" in class_name.lower()


def get_disease_info(class_name):
    if is_healthy(class_name):
        return HEALTHY_INFO
    return DISEASE_INFO.get(
        class_name,
        {
            "description": "Information for this condition is not available in the current database.",
            "treatment": "Consult a qualified agricultural professional for confirmation and treatment guidance.",
            "prevention": "Maintain good plant hygiene, airflow, and regular monitoring."
        }
    )


def image_quality(image):
    gray = image.convert("L")

    stat = ImageStat.Stat(gray)

    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    arr = np.array(gray, dtype=np.float32)

    if arr.shape[0] > 1 and arr.shape[1] > 1:
        dx = np.diff(arr, axis=1)
        dy = np.diff(arr, axis=0)
        sharpness = float(np.var(dx) + np.var(dy))
    else:
        sharpness = 0.0

    return brightness, contrast, sharpness


def predict_image(image):
    img = image.convert("RGB").resize((160, 160))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:5]

    top5 = [
        {
            "class": CLASS_NAMES[i],
            "name": pretty(CLASS_NAMES[i]),
            "probability": float(predictions[i])
        }
        for i in top_indices
    ]

    predicted_index = top_indices[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, top5


# ============================================================
# GRAD CAM
# ============================================================

def make_gradcam(image, predicted_index):
    try:
        base = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base = layer

        if base is None:
            return None

        last_conv_layer = None

        for layer in reversed(base.layers):
            if len(layer.output.shape) == 4:
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            [base.inputs],
            [last_conv_layer.output, base.output]
        )

        img = image.convert("RGB").resize((160, 160))
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, predicted_index]

        grads = tape.gradient(class_channel, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        if max_value > 0:
            heatmap /= max_value

        heatmap = heatmap.numpy()

        original = np.array(image.convert("RGB"))

        heatmap_img = Image.fromarray(
            np.uint8(heatmap * 255)
        ).resize(
            (original.shape[1], original.shape[0])
        )

        heatmap_arr = np.array(heatmap_img)

        plt.figure(figsize=(6, 5))
        plt.imshow(original)
        plt.imshow(heatmap_arr, alpha=0.45, cmap="jet")
        plt.axis("off")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()

        buffer.seek(0)

        return Image.open(buffer)

    except Exception:
        return None


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_report(prediction):
    disease_info = get_disease_info(prediction["class"])

    top5_html = ""

    for index, item in enumerate(prediction["top5"], start=1):
        top5_html += f"""
        <tr>
            <td>{index}</td>
            <td>{item["name"]}</td>
            <td>{item["probability"] * 100:.2f}%</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>PlantGuard AI Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}

            h1 {{
                font-size: 30px;
            }}

            h2 {{
                margin-top: 30px;
            }}

            .box {{
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 10px;
                margin-top: 15px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}

            th {{
                background: #f2f2f2;
            }}
        </style>
    </head>

    <body>

        <h1>🌱 PlantGuard AI</h1>
        <p>Plant Disease Detection Report</p>

        <div class="box">

            <h2>Prediction Summary</h2>

            <p><b>Plant:</b> {get_plant_name(prediction["class"])}</p>

            <p><b>Detected Condition:</b> {prediction["name"]}</p>

            <p><b>Confidence:</b> {prediction["confidence"] * 100:.2f}%</p>

            <p><b>Date & Time:</b> {prediction["timestamp"]}</p>

        </div>

        <div class="box">

            <h2>Description</h2>

            <p>{disease_info["description"]}</p>

        </div>

        <div class="box">

            <h2>Treatment / Management</h2>

            <p>{disease_info["treatment"]}</p>

        </div>

        <div class="box">

            <h2>Prevention</h2>

            <p>{disease_info["prevention"]}</p>

        </div>

        <div class="box">

            <h2>Top 5 Predictions</h2>

            <table>

                <tr>
                    <th>Rank</th>
                    <th>Condition</th>
                    <th>Probability</th>
                </tr>

                {top5_html}

            </table>

        </div>

        <br>

        <p>
        <b>Note:</b> This AI prediction is intended as a decision-support
        guide and should not replace confirmation by a qualified
        agricultural professional.
        </p>

    </body>
    </html>
    """

    return html


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 PlantGuard AI")

st.sidebar.write("AI-Powered Plant Disease Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Disease Detection",
        "🩺 Plant Health Guide",
        "📋 Prediction History",
        "📄 Disease Report",
        "🌿 Disease Library",
        "🤖 Model Information",
        "ℹ️ About Project"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌱 PlantGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Powered Plant Disease Detection and Health Guidance System</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Supported Classes</div>
                <div class="metric-value">{len(CLASS_NAMES)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="metric-title">Model</div>
                <div class="metric-value">MobileNetV2</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Predictions This Session</div>
                <div class="metric-value">{len(st.session_state.history)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">How PlantGuard AI Works</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("📷\n\n**1. Upload**\n\nUpload a clear leaf image.")

    with c2:
        st.info("🧠\n\n**2. Analyze**\n\nThe AI model analyzes the image.")

    with c3:
        st.info("🔬\n\n**3. Detect**\n\nThe system predicts the plant condition.")

    with c4:
        st.info("🩺\n\n**4. Guide**\n\nView treatment and prevention guidance.")

    st.markdown(
        '<div class="section-title">Supported Plants</div>',
        unsafe_allow_html=True
    )

    plants = sorted(
        list(set(get_plant_name(x) for x in CLASS_NAMES))
    )

    st.write(", ".join(plants))


# ============================================================
# DISEASE DETECTION
# ============================================================

elif page == "🔍 Disease Detection":

    st.markdown(
        '<div class="main-title">🔍 Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a single plant leaf image for AI-based disease prediction."
    )
    input_method = st.radio(
            "Choose image input method:",
            ["📁 Upload Image", "📷 Take Photo"],
            horizontal=True
        )

    if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a leaf image",
                type=["jpg", "jpeg", "png"]
            )
    else:
            uploaded_file = st.camera_input(
                "Take a clear photo of the plant leaf"
            )
    

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="section-title">Image Quality</div>',
                unsafe_allow_html=True
            )

            brightness, contrast, sharpness = image_quality(image)

            q1, q2, q3 = st.columns(3)

            q1.metric(
                "Brightness",
                f"{brightness:.1f}"
            )

            q2.metric(
                "Contrast",
                f"{contrast:.1f}"
            )

            q3.metric(
                "Sharpness",
                f"{sharpness:.1f}"
            )

        if st.button(
            "🔬 Analyze Leaf",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing the leaf..."):

                predicted_class, confidence, top5 = predict_image(image)

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            prediction = {
                "class": predicted_class,
                "name": pretty(predicted_class),
                "plant": get_plant_name(predicted_class),
                "confidence": confidence,
                "top5": top5,
                "timestamp": timestamp
            }

            st.session_state.last_prediction = prediction

            st.session_state.history.append(
                prediction
            )

            st.success("Analysis completed!")

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Plant",
                    get_plant_name(predicted_class)
                )

            with r2:
                st.metric(
                    "Condition",
                    pretty(predicted_class)
                )

            with r3:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            if confidence >= 0.80:
                st.success(
                    "High confidence prediction"
                )

            elif confidence >= 0.60:
                st.warning(
                    "Moderate confidence prediction"
                )

            else:
                st.error(
                    "Low confidence prediction — try a clearer image."
                )

            # ------------------------------------------------
            # HEALTH INFORMATION
            # ------------------------------------------------

            info = get_disease_info(predicted_class)

            st.markdown(
                '<div class="section-title">🩺 Health Guidance</div>',
                unsafe_allow_html=True
            )

            a, b = st.columns(2)

            with a:
                st.markdown("### 🔎 Description")
                st.write(info["description"])

                st.markdown("### 💊 Treatment / Management")
                st.write(info["treatment"])

            with b:
                st.markdown("### 🛡️ Prevention")
                st.write(info["prevention"])

            # ------------------------------------------------
            # TOP 5
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📊 Top 5 Predictions</div>',
                unsafe_allow_html=True
            )

            chart_df = pd.DataFrame({
                "Condition": [
                    item["name"] for item in top5
                ],
                "Probability": [
                    item["probability"] * 100
                    for item in top5
                ]
            })

            st.bar_chart(
                chart_df.set_index("Condition")
            )

            # ------------------------------------------------
            # GRAD CAM
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">🔥 AI Attention Map</div>',
                unsafe_allow_html=True
            )

            st.write(
                "The heatmap highlights image regions that contributed to the prediction."
            )

            predicted_index = CLASS_NAMES.index(
                predicted_class
            )

            heatmap = make_gradcam(
                image,
                predicted_index
            )

            if heatmap is not None:
                st.image(
                    heatmap,
                    caption="Grad-CAM Visualization",
                    use_container_width=True
                )
            else:
                st.info(
                    "Grad-CAM could not be generated for this model architecture."
                )

            # ------------------------------------------------
            # REPORT BUTTON
            # ------------------------------------------------

            report_html = generate_report(
                prediction
            )

            st.download_button(
                "📄 Download Disease Report",
                data=report_html,
                file_name="plantguard_disease_report.html",
                mime="text/html",
                use_container_width=True
            )


# ============================================================
# PLANT HEALTH GUIDE
# ============================================================

elif page == "🩺 Plant Health Guide":

    st.markdown(
        '<div class="main-title">🩺 Plant Health Guide</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select a condition to view its description, management and prevention guidance."
    )

    available_conditions = [
        x for x in CLASS_NAMES
        if x in DISEASE_INFO or is_healthy(x)
    ]

    selected = st.selectbox(
        "Select Plant Condition",
        available_conditions,
        format_func=pretty
    )

    info = get_disease_info(selected)

    st.markdown(
        f"## {pretty(selected)}"
    )

    st.markdown(
        f"**Plant:** {get_plant_name(selected)}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔎 Description")

        st.markdown(
            f"""
            <div class="info-box">
            {info["description"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 💊 Treatment / Management")

        st.markdown(
            f"""
            <div class="info-box">
            {info["treatment"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown("### 🛡️ Prevention")

        st.markdown(
            f"""
            <div class="info-box">
            {info["prevention"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📋 Prediction History":

    st.markdown(
        '<div class="main-title">📋 Prediction History</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No predictions have been made in this session yet."
        )

    else:

        total = len(st.session_state.history)

        average_confidence = np.mean([
            x["confidence"]
            for x in st.session_state.history
        ])

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "Total Predictions",
            total
        )

        h2.metric(
            "Average Confidence",
            f"{average_confidence * 100:.2f}%"
        )

        healthy_count = sum(
            1 for x in st.session_state.history
            if is_healthy(x["class"])
        )

        h3.metric(
            "Healthy Predictions",
            healthy_count
        )

        rows = []

        for item in st.session_state.history:

            rows.append({
                "Date & Time": item["timestamp"],
                "Plant": item["plant"],
                "Condition": item["name"],
                "Confidence": f"{item['confidence'] * 100:.2f}%"
            })

        history_df = pd.DataFrame(rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### 📊 Confidence Overview"
        )

        confidence_df = pd.DataFrame({
            "Prediction": range(
                1,
                len(st.session_state.history) + 1
            ),
            "Confidence": [
                x["confidence"] * 100
                for x in st.session_state.history
            ]
        })

        st.line_chart(
            confidence_df.set_index("Prediction")
        )

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.history = []

            st.success(
                "Prediction history cleared."
            )

            st.rerun()


# ============================================================
# DISEASE REPORT
# ============================================================

elif page == "📄 Disease Report":

    st.markdown(
        '<div class="main-title">📄 Disease Report</div>',
        unsafe_allow_html=True
    )

    prediction = st.session_state.last_prediction

    if prediction is None:

        st.info(
            "Make a prediction first from the Disease Detection page."
        )

    else:

        info = get_disease_info(
            prediction["class"]
        )

        st.markdown(
            "## 🌱 Latest Analysis"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Plant",
            prediction["plant"]
        )

        c2.metric(
            "Condition",
            prediction["name"]
        )

        c3.metric(
            "Confidence",
            f"{prediction['confidence'] * 100:.2f}%"
        )

        st.markdown("### 🔎 Description")

        st.write(
            info["description"]
        )

        st.markdown("### 💊 Treatment / Management")

        st.write(
            info["treatment"]
        )

        st.markdown("### 🛡️ Prevention")

        st.write(
            info["prevention"]
        )

        st.markdown("### 📊 Top 5 Predictions")

        report_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Condition": item["name"],
                "Probability": f"{item['probability'] * 100:.2f}%"
            }
            for i, item in enumerate(
                prediction["top5"]
            )
        ])

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        report_html = generate_report(
            prediction
        )

        st.download_button(
            "⬇️ Download Complete Report",
            data=report_html,
            file_name="plantguard_complete_report.html",
            mime="text/html",
            use_container_width=True
        )


# ============================================================
# DISEASE LIBRARY
# ============================================================

elif page == "🌿 Disease Library":

    st.markdown(
        '<div class="main-title">🌿 Disease Library</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Browse the plant conditions supported by the PlantGuard AI model."
    )

    search = st.text_input(
        "🔎 Search disease or plant"
    )

    filtered = CLASS_NAMES

    if search:

        filtered = [
            x for x in CLASS_NAMES
            if search.lower() in pretty(x).lower()
        ]

    st.write(
        f"Showing {len(filtered)} condition(s)"
    )

    for condition in filtered:

        info = get_disease_info(condition)

        with st.expander(
            pretty(condition)
        ):

            st.write(
                f"**Plant:** {get_plant_name(condition)}"
            )

            st.write(
                f"**Description:** {info['description']}"
            )

            st.write(
                f"**Treatment / Management:** {info['treatment']}"
            )

            st.write(
                f"**Prevention:** {info['prevention']}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.markdown(
        '<div class="main-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 🧠 Model Architecture

        **MobileNetV2**

        The application uses a MobileNetV2-based transfer-learning
        model for plant leaf classification.

        ### 📐 Input Size

        **160 × 160 pixels**

        ### 🌱 Number of Classes

        **38 PlantVillage classes**

        ### 🔬 Techniques Used

        - Transfer Learning
        - Image Classification
        - Confidence-based prediction
        - Top-5 probability analysis
        - Grad-CAM visualization
        - Image quality analysis

        ### 📊 Output

        The model produces a probability for each supported class.
        The class with the highest probability is displayed as the
        primary prediction.
        """
    )

    st.info(
        "The prediction should be treated as an AI-based decision-support result and not as a guaranteed expert diagnosis."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About PlantGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ## 🌱 Project Overview

        PlantGuard AI is an AI-powered plant disease detection
        application designed to identify plant leaf conditions from
        uploaded images.

        The system combines image classification with additional
        decision-support features to make the prediction easier to
        understand.

        ## 🎯 Project Objectives

        - Detect plant diseases from leaf images.
        - Provide confidence-based predictions.
        - Display the top five possible conditions.
        - Visualize important image regions using Grad-CAM.
        - Provide treatment and prevention guidance.
        - Maintain prediction history during the session.
        - Generate downloadable disease reports.

        ## 🛠️ Technologies

        - Python
        - TensorFlow
        - Keras
        - MobileNetV2
        - Streamlit
        - NumPy
        - Pandas
        - Matplotlib
        - PlantVillage dataset

        ## ⚠️ Important Note

        The system is intended as a project and decision-support
        application. Actual plant disease diagnosis should be
        confirmed using appropriate agricultural expertise.
        """
    )