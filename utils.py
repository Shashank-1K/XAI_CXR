import pywt
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from math import gcd


def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Chest X-ray Disease Prediction",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .success-text {
            color: #28a745;
            font-weight: bold;
        }
        .warning-text {
            color: #dc3545;
            font-weight: bold;
        }
        .info-box {
            background-color: #e8f4f8;
            border-left: 5px solid #4682B4;
            padding: 8px;
            border-radius: 5px;
            color : #005f99;
            font-size: 17px;
            font-weight: bold;
        }
        .navigation {
            background-color: #f1f1f1;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .step-complete {
            color: #28a745;
        }
        .step-current {
            font-weight: bold;
            color: #0D47A1;
        }
        .step-incomplete {
            color: #6c757d;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #6c757d;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

def create_card(title, content):
    """Create a styled card with title and content"""
    st.markdown(f"<div class='card'><h2 class='sub-header'>{title}</h2>{content}</div>", unsafe_allow_html=True)

def display_image_info(image):
    """Display basic information about an image"""
    if image is not None:
        h, w = image.shape[:2]
        info = f"**Image Size:** {w} x {h} pixels<br>"
        
        if len(image.shape) == 3:
            channels = image.shape[2]
            info += f"**Channels:** {channels}"
        else:
            info += "**Channels:** 1 (Grayscale)"
            
        return info
    return "No image information available."

def preprocess_image(image, method):
    """Apply basic preprocessing based on the selected feature method"""
    steps = []
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
        
    # Convert to grayscale (most feature extraction methods need grayscale)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    steps.append(("Grayscale Conversion", gray))

    gray_resized = cv2.resize(gray, (32, 32))
    steps.append(("Grayscale Conversion (32x32)", gray_resized))

    # Apply method-specific preprocessing
    
        
    if "Histogram" in method:
        # Create histogram visualization
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.figure(figsize=(5, 4))
        plt.plot(hist)
        plt.title("Image Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        # Convert plot to image
        fig = plt.gcf()
        plt.tight_layout()
        plt.close()
        
        steps.append(("Histogram Analysis", fig))
        
    elif "CNN" in method:
        # Simulating CNN preprocessing
        resized = cv2.resize(gray, (224, 224))
        steps.append(("Resized Image (224x224)", resized))
        
        # Simulating activation maps
        activation_map = np.random.rand(56, 56)
        plt.figure(figsize=(5, 4))
        plt.imshow(activation_map, cmap='viridis')
        plt.colorbar()
        plt.title("CNN Activation Map")
        
        fig = plt.gcf()
        plt.tight_layout()
        plt.close()
        
        steps.append(("Feature Maps", fig))
        
    # else:
    #     # Generic preprocessing steps
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     steps.append(("Gaussian Smoothing", blurred))
        
    #     # Apply thresholding
    #     _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    #     steps.append(("Thresholding", thresh))
    
    return steps

def format_probabilities(predictions):
    """Format disease predictions for display"""
    return [(name, f"{prob:.2%}") for name, prob in predictions]

def create_visualization(predictions):
    """Create visualization of disease probabilities"""
    classes = [p[0] for p in predictions]
    probs = [p[1] for p in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, probs, color='skyblue')
    
    # Highlight the most likely disease
    bars[0].set_color('navy')
    
    ax.set_xlabel('Probability')
    ax.set_title('Disease Prediction Probabilities')
    ax.invert_yaxis()  # To have the highest probability at the top
    plt.tight_layout()
    
    return fig

def get_feature_description(feature_method):
    """Get description for a feature extraction method"""
    method_descriptions = {
        "Histogram of Oriented Gradients (HOG)": """
            **HOG** calculates the distribution of gradient directions in the image, 
            which is effective for capturing shape features. It's particularly useful for 
            detecting structures like lung boundaries and nodules.
        """,
        "Local Binary Patterns (LBP)": """
            **LBP** analyzes texture patterns by comparing each pixel with its neighbors.
            It's effective for capturing fine texture details in lung tissue that may 
            indicate different pathologies.
        """,
        "Deep CNN Features (ResNet)": """
            **Deep CNN Features (ResNet)** leverage the ResNet architecture to extract high-level
            visual features, capturing complex patterns that can be highly diagnostic for
            many pulmonary conditions.
        """,
        "Deep CNN Features (VGG)": """
            **Deep CNN Features (VGG)** use the VGG architecture to capture hierarchical features
            from the X-ray image, which are effective for identifying subtle patterns
            associated with different lung diseases.
        """,
        "Edge Detection Features": """
            **Edge Detection Features** highlight structures and boundaries in the X-ray image,
            which can reveal abnormalities in lung shape, size, and structure.
        """
    }
    
    # Return description if available, otherwise generate a generic one
    if feature_method in method_descriptions:
        return method_descriptions[feature_method]
    
    # Generate generic description based on keywords
    if "Edge" in feature_method:
        return f"""
            **{feature_method}** detects edges and transitions in the X-ray image,
            highlighting structures and boundaries that may indicate abnormalities.
        """
    elif "Histogram" in feature_method:
        return f"""
            **{feature_method}** analyzes the distribution of intensity values,
            which can reveal patterns associated with various lung conditions.
        """
    elif "CNN" in feature_method:
        model_type = feature_method.split("(")[1].replace(")", "")
        return f"""
            **{feature_method}** leverages the {model_type} architecture to extract high-level
            visual features, capturing complex patterns that can be highly diagnostic for
            many pulmonary conditions.
        """
    else:
        return f"""
            **{feature_method}** extracts distinctive features from the X-ray image
            that are useful for identifying patterns associated with different pathologies.
        """
def rdeigValues(image):            
    rdeig = [np.linalg.matrix_rank(image)]
    rdeig.append(np.linalg.det(image))
    rdeig.append(image.trace())
    eva, evec = np.linalg.eig(image)
    eva = sorted(eva, reverse=True)
    rdeig +=[np.abs(i) if np.iscomplex(i) else i for i in eva]
    return rdeig   

def generate_wavelet_scalogram(image):
    coefficients, frequencies = pywt.cwt(image, np.arange(1, 32), 'gaus1')
    scalogram = np.abs(coefficients)
    scalogram_resized = cv2.resize(scalogram[0], (32, 32))
    return scalogram_resized

def lcm(x, y):
    return abs(x * y) // gcd(x, y) if x and y else 0

def smith_normal_form(matrix):
    A = np.array(matrix, dtype=int)
    m, n = A.shape
    S = np.eye(m, dtype=int)
    T = np.eye(n, dtype=int)
    i = 0
    j = 0
    while i < m and j < n:
        min_val = np.inf
        min_pos = None
        for p in range(i, m):
            for q in range(j, n):
                if A[p, q] != 0 and abs(A[p, q]) < min_val:
                    min_val = abs(A[p, q])
                    min_pos = (p, q)
        if min_pos is None:
            i += 1
            j += 1
            continue
        p, q = min_pos
        A[[i, p]] = A[[p, i]]
        S[[i, p]] = S[[p, i]] 
        A[:, [j, q]] = A[:, [q, j]]
        T[:, [j, q]] = T[:, [q, j]]  
        for p in range(i + 1, m):
            if A[p, j] != 0:
                g = gcd(A[i, j], A[p, j])
                lcm_val = lcm(A[i, j], A[p, j])
                a_i = lcm_val // A[i, j]
                a_p = lcm_val // A[p, j]
                
                A[p, :] = a_p * A[p, :] - a_i * A[i, :]
                S[p, :] = a_p * S[p, :] - a_i * S[i, :]
        for q in range(j + 1, n):
            if A[i, q] != 0:
                g = gcd(A[i, j], A[i, q])
                lcm_val = lcm(A[i, j], A[i, q])
                a_i = lcm_val // A[i, j]
                a_q = lcm_val // A[i, q]

                A[:, q] = a_q * A[:, q] - a_i * A[:, j]
                T[:, q] = a_q * T[:, q] - a_i * T[:, j]
        A[i, j] = abs(A[i, j])
        for p in range(i):
            if A[p, j] != 0:
                g = gcd(A[i, j], A[p, j])
                A[p, :] = A[p, :] - (A[p, j] // g) * A[i, :]
                S[p, :] = S[p, :] - (A[p, j] // g) * S[i, :]
        
        for q in range(j):
            if A[i, q] != 0:
                g = gcd(A[i, j], A[i, q])
                A[:, q] = A[:, q] - (A[i, q] // g) * A[:, j]
                T[:, q] = T[:, q] - (A[i, q] // g) * T[:, j]

        i += 1
        j += 1
    A_diag = np.diag(np.diag(A))
    return np.diag(A_diag)


def compute_smith_normal_form(matrix, window_size=5):
    A = np.array(matrix, dtype=int)
    m, n = A.shape
    results = np.zeros((m - window_size + 1, n - window_size + 1, window_size))

    # Iterate over all possible submatrices of the given window size
    for i in range(m - window_size + 1):
        for j in range(n - window_size + 1):
            submatrix = A[i:i+window_size, j:j+window_size]
            results[i, j, :] = smith_normal_form(submatrix)
    
    return results

def feature_extraction(image, method):
    # Convert RGB to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize to 32x32
    gray_resized = cv2.resize(gray, (32, 32))
    feature_vector=[]
    if "Raw Pixel Values" == method:
        # Flatten the image into a 1D vector
        feature_vector.extend(gray_resized.flatten().astype(np.float32))
    elif "Matrix Properties (Pixel,Rank,Det,Trace)" == method:
        feature_vector.extend(gray_resized.flatten().astype(np.float32))
        feature_vector.extend(rdeigValues(gray_resized))
    elif "Matrix Properties (Rank,Det,Trace)" == method:
        feature_vector.extend(rdeigValues(gray_resized))
    elif "Pixels and MPs of Scalograms(CWT,STFT)" == method:
        image_resized = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        feature_vector.extend(image_gray.flatten().astype(np.float32))
        cwt_scalogram = generate_wavelet_scalogram(image_gray)
        scalogram_normalized = cv2.normalize(cwt_scalogram, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating CWT Scalogram",image_color))
        cwt_rdeig = rdeigValues(cwt_scalogram)
        feature_vector.extend(cwt_rdeig)
        f, t, Zxx = signal.stft(image_gray, fs=1, nperseg=32)
        stft_resized = np.abs(Zxx)
        stft_resized_resized = cv2.resize(stft_resized, (32, 32))
        image_gray1 = cv2.cvtColor(stft_resized_resized, cv2.COLOR_BGR2GRAY)
        scalogram_normalized = cv2.normalize(stft_resized_resized, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating STFT Scalogram",image_color))
        stft_rdeig = rdeigValues(image_gray1)
        feature_vector.extend(stft_rdeig)
    elif "MPs of Original,CWT,STFT" == method:
        image_resized = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        feature_vector.extend(rdeigValues(gray_resized))
        cwt_scalogram = generate_wavelet_scalogram(image_gray)
        scalogram_normalized = cv2.normalize(cwt_scalogram, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)    
        st.session_state.processing_visualizations.append((f"Generating CWT Scalogram",image_color))
        cwt_rdeig = rdeigValues(cwt_scalogram)
        feature_vector.extend(cwt_rdeig)
        f, t, Zxx = signal.stft(image_gray, fs=1, nperseg=32)
        stft_resized = np.abs(Zxx)
        stft_resized_resized = cv2.resize(stft_resized, (32, 32))
        image_gray1 = cv2.cvtColor(stft_resized_resized, cv2.COLOR_BGR2GRAY)
        scalogram_normalized = cv2.normalize(image_gray1, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating STFT Scalogram",image_color))
        stft_rdeig = rdeigValues(image_gray1)
        feature_vector.extend(stft_rdeig)
    elif "Pixels of Original and MPs of Original,CWT,STFT" == method:
        image_resized = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        feature_vector.extend(gray_resized.flatten().astype(np.float32))
        feature_vector.extend(rdeigValues(gray_resized))
        cwt_scalogram = generate_wavelet_scalogram(image_gray)
        scalogram_normalized = cv2.normalize(cwt_scalogram, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating CWT Scalogram",image_color))
        cwt_rdeig = rdeigValues(cwt_scalogram)
        feature_vector.extend(cwt_rdeig)
        f, t, Zxx = signal.stft(image_gray, fs=1, nperseg=32)
        stft_resized = np.abs(Zxx)
        stft_resized_resized = cv2.resize(stft_resized, (32, 32))
        image_gray1 = cv2.cvtColor(stft_resized_resized, cv2.COLOR_BGR2GRAY)
        scalogram_normalized = cv2.normalize(image_gray1, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating STFT Scalogram",image_color))
        stft_rdeig = rdeigValues(image_gray1)
        feature_vector.extend(stft_rdeig)
    elif "Pixels of Original,CWT,STFT and MPs of Original,CWT,STFT" == method:
        image_resized = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        feature_vector.extend(gray_resized.flatten().astype(np.float32))
        feature_vector.extend(rdeigValues(gray_resized))
        cwt_scalogram = generate_wavelet_scalogram(image_gray)
        scalogram_normalized = cv2.normalize(cwt_scalogram, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating CWT Scalogram",image_color))
        feature_vector.extend(cwt_scalogram.flatten().astype(np.float32))
        cwt_rdeig = rdeigValues(cwt_scalogram)
        feature_vector.extend(cwt_rdeig)
        f, t, Zxx = signal.stft(image_gray, fs=1, nperseg=32)
        stft_resized = np.abs(Zxx)
        stft_resized_resized = cv2.resize(stft_resized, (32, 32))
        image_gray1 = cv2.cvtColor(stft_resized_resized, cv2.COLOR_BGR2GRAY)
        scalogram_normalized = cv2.normalize(image_gray1, None, 0, 255, cv2.NORM_MINMAX)
        scalogram_uint8 = scalogram_normalized.astype(np.uint8)
        image_color = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_JET)
        st.session_state.processing_visualizations.append((f"Generating STFT Scalogram",image_color))
        feature_vector.extend(image_gray1.flatten().astype(np.float32))
        stft_rdeig = rdeigValues(image_gray1)
        feature_vector.extend(stft_rdeig)
    elif "Smith Normal Form With Window Size 5" == method:
        image_resized = cv2.resize(image, (32, 32))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        feature_vector = compute_smith_normal_form(image_gray,5)
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
        flattened_data = feature_vector.reshape(feature_vector.shape[0], -1)
        flattened_data = feature_vector.reshape(1, -1)
        feature_vector=flattened_data
        feature_vector = feature_vector.reshape(1, 3920)
    return feature_vector

def quant(feature_vector):
    transformer = QuantileTransformer(random_state=0)
    features = np.array(feature_vector).astype(np.float32)
    features[np.isinf(features)] = 255
    features1 = [[i] for i in features]
    features1 = transformer.fit_transform(features1)
    feature_vector = [np.abs(i) if np.iscomplex(i) else i for i in features1]
    return feature_vector

def minmax_normalize(feature_vector):
    scaler = MinMaxScaler()
    features = np.array(feature_vector).astype(np.float32)
    features[np.isinf(features)] = 255  # Replace infinite values
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = scaler.fit_transform(features)
    # feature_vector = [np.abs(i) if np.iscomplex(i) else i for i in features]
    if np.iscomplexobj(features):
        features = np.abs(features)
    return feature_vector

def standardize(feature_vector):
    scaler = StandardScaler()
    features = np.array(feature_vector).astype(np.float32)
    features[np.isinf(features)] = 255  # Replace infinite values
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = scaler.fit_transform(features)
    # feature_vector = [np.abs(i) if np.iscomplex(i) else i for i in features]
    if np.iscomplexobj(features):
        features = np.abs(features)
    return feature_vector

def feature_normalisation(feature_vector,method):
    classification = st.session_state.selected_classification 
    if "Raw Pixel Values" == method:
        if classification == "CN":
            feature_vector=minmax_normalize(feature_vector)
        elif classification == "NP":
            feature_vector=standardize(feature_vector)
        elif classification == "CP":
            feature_vector=quant(feature_vector)
        elif classification == "CNP":
            feature_vector=quant(feature_vector)
    elif "Matrix Properties (Pixel,Rank,Det,Trace)" == method:
        if classification == "CN":
            feature_vector=minmax_normalize(feature_vector)
        elif classification == "NP":
            feature_vector=minmax_normalize(feature_vector)
        elif classification == "CP":
            feature_vector=quant(feature_vector)
        elif classification == "CNP":
            feature_vector=quant(feature_vector)
    elif "Matrix Properties (Rank,Det,Trace)" == method:
        if classification == "CN":
            feature_vector=standardize(feature_vector)
        elif classification == "NP":
            feature_vector=quant(feature_vector)
        elif classification == "CP":
            feature_vector=(feature_vector)
        elif classification == "CNP":
            feature_vector=quant(feature_vector)
    elif "Pixels and MPs of Scalograms(CWT,STFT)" == method:
        feature_vector=quant(feature_vector)
    elif "MPs of Original,CWT,STFT" in method:
        feature_vector=quant(feature_vector)
    elif "Pixels of Original and MPs of Original,CWT,STFT" == method:
        if classification == "NP":
            feature_vector=minmax_normalize(feature_vector)
        else:
            feature_vector=quant(feature_vector)
    elif "Pixels of Original,CWT,STFT and MPs of Original,CWT,STFT" == method:
        feature_vector=quant(feature_vector)
    elif "Smith Normal Form With Window Size 5" == method:
        if classification == "NP":
            feature_vector=minmax_normalize(feature_vector)
        else:
            feature_vector=standardize(feature_vector)
    return feature_vector


def svd_visualisation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Preprocess the image
    image_32x32 = cv2.resize(gray_image, (224, 224))
    
    # Step 1: Perform SVD
    U, S, VT = np.linalg.svd(image_32x32, full_matrices=True)
    
    # Step 2: Retain top eigenvalue
    S_new = np.zeros_like(S)
    S_new[0] = S[0]  # Retain only the largest singular value
    S_matrix = np.diag(S_new[:1])  # Create a 1x1 diagonal matrix
    
    # Step 3: Modify U and VT
    U_new = U[:, :1]  # Retain the first column of U (32x1)
    VT_new = VT[:1, :]  # Retain the first row of V^T (1x32)
    
    # Step 4: Reconstruct the approximation
    A_approx = np.dot(U_new, np.dot(S_matrix, VT_new))
    
    # Step 5: Resize the reconstructed image to 224x224
    A_approx_224 = cv2.resize(A_approx, (224, 224))
    
    # Step 6: Compute the difference
    difference = np.abs(image_32x32 - A_approx)
    difference_224 = cv2.resize(difference, (224, 224))
    
    # Normalize the difference for heatmap
    normalized_diff = (difference_224 - np.min(difference_224)) / (np.max(difference_224) - np.min(difference_224))
    
    # Convert normalized difference to heatmap
    heatmap = cv2.applyColorMap((normalized_diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the reconstructed image
    overlay = cv2.addWeighted(cv2.cvtColor(A_approx_224.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    

    from matplotlib.colors import Normalize
    # Normalize for consistent visualization
    norm = Normalize(vmin=0, vmax=255)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # return overlay

    return overlay


# Define the path to the prototypes folder
prototypes_path = "./prototypes"

# Load the prototypes corresponding to each class
covid_prot = np.load(os.path.join(prototypes_path, 'class_covid_prototype.npy'))
normal_prot = np.load(os.path.join(prototypes_path, 'class_normal_prototype.npy'))
pneumonia_prot = np.load(os.path.join(prototypes_path, 'class_pneumonia_prototype.npy'))

# Function to assign the prototype based on predicted class
def get_prototype(predicted_class):
    """Returns the appropriate prototype based on the predicted class"""
    if predicted_class[0] == 'Normal':
        return normal_prot
    elif predicted_class[0] == 'COVID-19':
        return covid_prot
    elif predicted_class[0] == 'Pneumonia':
        return pneumonia_prot
    else:
        print(predicted_class)
        raise ValueError("Unknown predicted class: " + predicted_class)

# Example usage of the function



# Print the selected prototype shape for verification print(f"Selected Prototype for {predicted_class}: {selected_prototype.shape}")


def Pattern_Visualization(test_image):
    """
    Process an image, highlight the differences with the prototype, and visualize the results.
    """
    # predicted_class=st.session_state.label
    predicted_class = st.session_state.label
    prototype = get_prototype(predicted_class)
    def px_process_image(image):
        features = []
        # Resize the image
        image_resized = cv2.resize(image, (224, 224))
        
        # Convert to grayscale
        if len(image_resized.shape) == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_resized
        
        # Flatten the image
        features.extend(image_gray.flatten())
        
        # Scale the features using QuantileTransformer
        transformer = QuantileTransformer(random_state=0)
        features = np.array(features).astype(np.float32)
        features[np.isinf(features)] = 255  # Handle infinities
        features_scaled = transformer.fit_transform(features.reshape(-1, 1))
        return features_scaled.flatten()

    def highlight_diff(test_features, prototype):
        """
        Calculate and visualize differences between the test features and the prototype.
        """
        # Separate pixel-level features from other features
        pixel_features = test_features[:50176]  # First 224x224 = 50176 values
        prototype_pixel_features = prototype[:50176]

        # Compute absolute differences
        diff_pixel_features = np.abs(pixel_features - prototype_pixel_features)

        # Normalize the differences
        diff_pixel_normalized = cv2.normalize(diff_pixel_features, None, 0, 255, cv2.NORM_MINMAX)

        # Reshape to image dimensions
        diff_pixel_image = diff_pixel_normalized.reshape(224, 224)

        # Create a heatmap
        heatmap_pixel = cv2.applyColorMap(diff_pixel_image.astype(np.uint8), cv2.COLORMAP_JET)

        return heatmap_pixel, diff_pixel_image

    # Check if the image path exists
    #if not os.path.exists(orgimg_path):
       # raise FileNotFoundError(f"The image file does not exist at: {orgimg_path}")


    # Process the image to extract features
    px_test_features = px_process_image(test_image)

    # Fetch the prototype for the predicted class
    # predicted_class_name = [key for key, value in label_dict.items() if value == predicted_class]
    # if not predicted_class_name:
        # raise ValueError(f"Invalid predicted class: {predicted_class}")
    # predicted_class_name = predicted_class_name[0]

    # prototype = prototypes.get(predicted_class_name)
    if prototype is None:
        raise ValueError(f"No prototype found for class")

    # Highlight the differences between the test image and its prototype
    highlighted_image, matrix_diff_image = highlight_diff(px_test_features, prototype)

    # Prepare for visualization
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Resize the highlighted image to match the original image size
    highlighted_image_resized = cv2.resize(highlighted_image, (test_image.shape[1], test_image.shape[0]))

    # Convert to BGR if necessary
    if len(highlighted_image_resized.shape) == 2:
        highlighted_image_resized = cv2.cvtColor(highlighted_image_resized, cv2.COLOR_GRAY2BGR)

    # Overlay the heatmap on the original image
    alpha = 0.4
    overlay = cv2.addWeighted(test_image_rgb, 1 - alpha, highlighted_image_resized, alpha, 0)

    from matplotlib.colors import Normalize

    # Normalize for consistent visualization
    norm = Normalize(vmin=0, vmax=255)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay

    # return overlay